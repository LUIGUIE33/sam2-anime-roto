# -*- coding: utf-8 -*-
"""
Anime Roto Test Runner (en español)
Flujo: elegir clip -> extraer frames -> seleccionar sujeto -> rastrear ->
revisar/corregir -> vista previa -> exportar para After Effects.
"""
import os
import sys
import json
import queue
import shutil
import threading
from pathlib import Path

import numpy as np

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk
import sam2_video_tool as tool

MODELOS = {
    "Rápido (Tiny)": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml",
                      "Rápido. Menos finura en bordes finos."),
    "Equilibrado (Base+)": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml",
                            "Balance equilibrado entre velocidad y precisión."),
    "Calidad (Large)": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml",
                        "Máxima precisión. Requiere ~4 GB de VRAM."),
}
DATOS = BASE / "anime_roto_data"
CHECKPOINTS = BASE / "sam2-main" / "checkpoints"
CONFIGS = BASE / "sam2-main" / "sam2"

MASCARA_COLOR = (255, 130, 60)  # BGR azulado para resaltar la selección


def _compone(mascara, frame):
    """Superpone la máscara (bool) sobre el frame con tinte azulado."""
    vis = frame.copy()
    vis[mascara] = (vis[mascara].astype(np.float32) * 0.55 +
                    np.array(MASCARA_COLOR) * 0.45).astype(np.uint8)
    return vis


def _id_archivo(nombre):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in nombre)


class RotoApp:
    def __init__(self, root):
        self.root = root
        root.title("Anime Roto Test Runner")
        ancho_pantalla = root.winfo_screenwidth()
        alto_pantalla = root.winfo_screenheight()
        ancho_v = min(820, ancho_pantalla - 40)
        alto_v = min(720, alto_pantalla - 80)
        root.geometry(f"{ancho_v}x{alto_v}")
        root.minsize(740, 600)

        self.video = None
        self.clip_dir = None
        self.meta = {}
        self.n_frames = 0
        self.fps = 30.0
        self.pred = None
        self.pred_modelo = None
        self.cola = queue.Queue()
        self.ocupado = False

        self.var_modelo = tk.StringVar(value=list(MODELOS.keys())[0])
        self.var_estado = tk.StringVar(value="")
        self.var_desc = tk.StringVar(value=MODELOS[self.var_modelo.get()][2])

        self._construir_ui()
        self._actualizar_estado()
        self.root.after(120, self._poll_cola)

    # ---------------------------------------------------------- UI
    def _construir_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)
        pad = {"padx": 4, "pady": 3}

        frm.head = ttk.Frame(frm)
        frm.head.pack(fill="x")
        ttk.Label(frm.head,
                  text="Elige un clip, extrae los frames y luego selecciona el sujeto en el frame 1.",
                  wraplength=740).pack(anchor="w")
        self.lbl_video = ttk.Label(frm.head, text="Video seleccionado: (ninguno)",
                                   foreground="#202020", wraplength=740)
        self.lbl_video.pack(anchor="w", pady=(6, 0))
        self.lbl_estado = ttk.Label(frm.head, textvariable=self.var_estado,
                                    foreground="#0a6b3d", wraplength=740)
        self.lbl_estado.pack(anchor="w")

        frm.modelo = ttk.LabelFrame(frm, text="Modelo", padding=8)
        frm.modelo.pack(fill="x", pady=(10, 0))
        row = ttk.Frame(frm.modelo)
        row.pack(fill="x")
        self.cb_modelo = ttk.Combobox(row, textvariable=self.var_modelo,
                                      values=list(MODELOS.keys()),
                                      state="readonly", width=24)
        self.cb_modelo.pack(side="left")
        self.cb_modelo.bind("<<ComboboxSelected>>", self._sel_modelo)
        ttk.Label(frm.modelo, textvariable=self.var_desc, foreground="#707070",
                  wraplength=600).pack(anchor="w")
        ttk.Label(frm.modelo,
                  text="El reconocimiento corre en GPU (RTX). Para los clips usados en el prototipo, "
                       "Equilibrado (Base+) es la opción más probada. Rápido (Tiny) va más deprisa; "
                       "Calidad (Large) da los bordes más finos.", foreground="#909090",
                  wraplength=760).pack(anchor="w", pady=(6, 0))

        frm.btns = ttk.LabelFrame(frm, text="Pasos", padding=8)
        frm.btns.pack(fill="x", pady=(10, 0))
        g = ttk.Frame(frm.btns)
        g.pack(fill="x")
        fila = 0
        columna = 0
        nombres = ("elegir extraer abrirframes seleccionar editar rastrear "
                   "abrirmascaras revisar preview abrirpreview exportar limpiar "
                   "roto abrirroto abrirresultados").split()
        for texto, callback in [
            ("Elegir video", self.elegir_video),
            ("Extraer frames", self.extraer_frames),
            ("Abrir carpeta de frames", lambda: self._abrir(self.frames()),),
            ("Seleccionar sujeto en frame 1", self.seleccionar_sujeto),
            ("Editar máscara seleccionada", self.editar_mascara),
            ("Rastrear clip", self.rastrear),
            ("Abrir carpeta de máscaras", lambda: self._abrir(self.masks()),),
            ("Revisar y corregir frames", self.revisar),
            ("Crear video de vista previa", self.crear_preview),
            ("Abrir vista previa", self.abrir_preview),
            ("Exportar máscaras para After Effects", self.exportar_ae),
            ("Limpiar carpetas de máscaras antiguas", self.limpiar_viejo),
            ("Exportar rotoscopia (sin fondo)", self.exportar_roto),
            ("Abrir rotoscopia", self.abrir_roto),
            ("Abrir carpeta de resultados", self.abrir_resultados),
        ]:
            b = ttk.Button(g, text=texto, command=callback, width=34)
            b.grid(row=fila, column=columna, sticky="ew", **pad)
            setattr(self, f"btn_{nombres.pop(0)}", b)
            columna += 1
            if columna > 1:
                columna = 0
                fila += 1
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=1)

        self.lbl_paso = ttk.Label(frm, text="", foreground="#1a5a8a")
        self.lbl_paso.pack(fill="x", pady=(6, 0))

        frm.prog = ttk.Frame(frm)
        frm.prog.pack(fill="x", pady=(12, 0))
        self.lbl_prog = ttk.Label(frm.prog, text="")
        self.lbl_prog.pack(anchor="w")
        self.barra = ttk.Progressbar(frm.prog, maximum=100)
        self.barra.pack(fill="x", pady=(4, 0))

        frm.log = ttk.LabelFrame(frm, text="Registro")
        frm.log.pack(fill="both", expand=True, pady=(10, 0))
        self.txt_log = tk.Text(frm.log, height=8, font=("Consolas", 9), state="disabled",
                               wrap="none")
        self.txt_log.pack(fill="both", expand=True)

    def _log(self, msj):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msj + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    def _set_estado(self, texto):
        self.var_estado.set(texto)

    # ---------------------------------------------------------- modelos y utilidades
    def _sel_modelo(self, *a):
        self.var_desc.set(MODELOS[self.var_modelo.get()][2])

    def _ruta_modelo(self):
        pt, cfg, _ = MODELOS[self.var_modelo.get()]
        return CHECKPOINTS / pt, CONFIGS / cfg

    def _frames(self):
        return self.clip_dir / "frames"

    def _masks(self):
        return self.clip_dir / "masks"

    def _ae(self):
        return self.clip_dir / "ae_export"

    def _preview(self):
        return self.clip_dir / "preview"

    def _ruta_roto(self):
        if not self.clip_dir:
            return None
        return self.clip_dir / f"{_id_archivo(self.video.stem)}_rotoscopia.mov"

    def _abrir(self, carpeta):
        if carpeta and carpeta.exists():
            os.startfile(str(carpeta))

    def _guardar_meta(self):
        if self.meta is not None:
            (self.clip_dir / "meta.json").write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _actualizar_estado(self):
        frames_ok = self.n_frames > 0
        masks_ok = (self.meta.get("rastreado") and self.masks().exists()
                    and any(self.masks().glob("mask_*.npy")))
        sel_ok = bool(self.meta.get("punto") or self.meta.get("box"))
        st = {}
        st["elegir"] = not self.ocupado
        st["extraer"] = self.video is not None and not self.ocupado
        st["abrirframes"] = frames_ok and not self.ocupado
        st["seleccionar"] = frames_ok and not self.ocupado
        st["editar"] = masks_ok and not self.ocupado
        st["rastrear"] = frames_ok and sel_ok and not self.ocupado
        st["abrirmascaras"] = masks_ok and not self.ocupado
        st["revisar"] = masks_ok and not self.ocupado
        st["preview"] = masks_ok and not self.ocupado
        st["abrirpreview"] = (self.clip_dir is not None
                              and (self._preview() / "preview.mp4").exists()
                              and not self.ocupado)
        st["roto"] = masks_ok and not self.ocupado
        st["abrirroto"] = (self._ruta_roto() is not None
                           and self._ruta_roto().exists()
                           and not self.ocupado)
        st["abrirresultados"] = (self.clip_dir is not None
                                 and self.clip_dir.exists()
                                 and not self.ocupado)
        st["exportar"] = masks_ok and not self.ocupado
        for nombre, estado in st.items():
            getattr(self, f"btn_{nombre}").config(state="normal" if estado else "disabled")

        if self.video is None:
            paso = ("Paso 1 de 6: pulsa 'Elegir video' y escoge tu clip. "
                    "El resto de botones se activa al avanzar.")
        elif self.n_frames == 0:
            paso = "Paso 2 de 6: pulsa 'Extraer frames' para convertir el clip en imágenes."
        elif not (self.meta.get("punto") or self.meta.get("box")):
            paso = ("Paso 3 de 6: pulsa 'Seleccionar sujeto en frame 1'. En esa ventana, "
                    "haz clic en la persona — se marcará en azul — y pulsa Guardar.")
        elif not self.meta.get("rastreado"):
            paso = "Paso 4 de 6: pulsa 'Rastrear clip' para el tracking completo de frames."
        else:
            paso = ("Rastreo listo. Paso 5: 'Revisar y corregir frames' o 'Crear video de vista previa'. "
                    "Paso 6: 'Exportar rotoscopia (sin fondo)' o 'Exportar máscaras para After Effects'.")
        self.lbl_paso.config(text=paso)

    # ---------------------------------------------------------- pasos
    def elegir_video(self):
        ruta = filedialog.askopenfilename(
            title="Elegir clip",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("Todos", "*.*")])
        if not ruta:
            return
        self.video = Path(ruta)
        self.n_frames = 0
        self.clip_dir = DATOS / _id_archivo(self.video.stem)
        self.meta = {"video": str(self.video), "modelo": self.var_modelo.get(),
                     "punto": None, "box": None, "correcciones": [], "rastreado": False}
        if (self.clip_dir / "meta.json").exists():
            try:
                self.meta.update(json.loads((self.clip_dir / "meta.json").read_text(encoding="utf-8")))
                self.n_frames = len(list((self.clip_dir / "frames").glob("*.jpg")))
            except Exception:
                pass
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_video.config(text=f"Video seleccionado: {self.video}")
        self._log(f"Clip elegido: {self.video}")
        self._set_estado("Elige un clip, extrae los frames y luego selecciona el sujeto en el frame 1.")
        self._guardar_meta()
        self._actualizar_estado()

    def extraer_frames(self):
        if self.ocupado or not self.video:
            return
        carpeta = self._frames()
        if carpeta.exists():
            if not messagebox.askyesno("Extraer frames",
                                       "¿Volver a extraer los frames de este clip? Se borrarán los actuales."):
                return
            shutil.rmtree(carpeta, ignore_errors=True)
        carpeta.mkdir(parents=True, exist_ok=True)
        self.ocupado = True
        self._actualizar_estado()

        def trabajo():
            try:
                self.fps = tool.extraer_frames(str(self.video), str(carpeta),
                                               progreso=self._prog)
                n = len(list(carpeta.glob("*.jpg")))
                self.cola.put(("hecho", True, f"Frames extraídos: {n} ({self.fps:.2f} fps)"))
                self.cola.put(("n_frames", n))
            except Exception as e:
                self.cola.put(("hecho", False, f"Error al extraer frames: {e}"))

        threading.Thread(target=trabajo, daemon=True).start()

    def _frames_primer(self):
        import glob
        archivos = sorted(glob.glob(str(self._frames() / "*.jpg")))
        return archivos[0] if archivos else None

    @property
    def frames(self):
        return self._frames

    @property
    def masks(self):
        return self._masks

    def seleccionar_sujeto(self):
        if self.n_frames <= 0:
            return
        SelWindow(self, "editar" if self.meta.get("rastreado") else "seleccionar")

    def editar_mascara(self):
        if not (self.meta.get("rastreado") and self.masks().exists()):
            messagebox.showinfo("Editar máscara", "Primero rastrea el clip con 'Rastrear clip'.")
            return
        SelWindow(self, "editar")

    def rastrear(self):
        if self.ocupado or self.n_frames <= 0:
            return
        if not (self.meta.get("punto") or self.meta.get("box")):
            messagebox.showinfo("Rastrear clip",
                                "Selecciona primero el sujeto (botón 'Seleccionar sujeto en frame 1').")
            return
        cp, cfg = self._ruta_modelo()
        if not cp.exists() or not cfg.exists():
            messagebox.showerror("Modelo no disponible",
                                 f"El checkpoint o la config de '{self.var_modelo.get()}' no existen.\n\n"
                                 f"Falta: {cp if not cp.exists() else cfg}")
            return
        if self.pred_modelo != self.var_modelo.get() or self.pred is None:
            self.ocupado = True
            self._actualizar_estado()
            self._set_estado(f"Cargando modelo {self.var_modelo.get()}...")

            def cargar():
                try:
                    self.pred = tool.cargar_predictor(str(cp), str(cfg))
                    self.pred_modelo = self.var_modelo.get()
                    self.cola.put(("pred_listo", self.var_modelo.get()))
                except Exception as e:
                    self.cola.put(("hecho", False, f"Error de carga del modelo: {e}"))
            threading.Thread(target=cargar, daemon=True).start()
            return
        self._lanzar_rastreo(self.pred)

    def _lanzar_rastreo(self, pred):
        if self.masks().exists() and any(self.masks().glob("mask_*.npy")):
            if not messagebox.askyesno("Rastrear clip",
                                       "Ya hay máscaras de este clip. ¿Quieres rastrearlo de nuevo "
                                       "(las actuales serán reemplazadas)?"):
                return
            shutil.rmtree(self.masks(), ignore_errors=True)
        self.masks().mkdir(parents=True, exist_ok=True)
        self.ocupado = True
        self._actualizar_estado()
        self._set_estado("Rastreando clip...")

        def trabajo():
            try:
                punto = tuple(self.meta["punto"]) if self.meta.get("punto") else None
                box = tuple(self.meta["box"]) if self.meta.get("box") else None
                extra = [(x, y, et) for x, y, et in self.meta.get("correcciones", [])]
                mascaras = tool.segmentar_video(pred, str(self.frames()),
                                                punto_xy=punto, box_xyxy=box,
                                                puntos_extra=extra or None,
                                                progreso=self._prog)
                for idx, m in mascaras.items():
                    np.save(self.masks() / f"mask_{idx:05d}.npy", m)
                self.meta["rastreado"] = True
                self.meta["modelo"] = self.var_modelo.get()
                self._guardar_meta()
                self.cola.put(("hecho", True,
                               f"Máscaras listas ({len(mascaras)} frames). Revisa con 'Revisar y corregir frames'."))
            except Exception as e:
                import traceback
                self.cola.put(("hecho", False, f"Error al rastrear:\n{traceback.format_exc()[-1200:]}"))

        threading.Thread(target=trabajo, daemon=True).start()

    def revisar(self):
        if not self.meta.get("rastreado"):
            messagebox.showinfo("Revisar", "Primero rastrea el clip.")
            return
        ReviewWindow(self)

    def crear_preview(self, *a):
        if self.ocupado or not self.meta.get("rastreado"):
            return
        pre = self._preview()
        pre.mkdir(parents=True, exist_ok=True)
        tmp = pre / "tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.ocupado = True
        self._actualizar_estado()
        self._set_estado("Generando vista previa...")

        def trabajo():
            try:
                n = 0
                archivos = sorted(self.frames().glob("*.jpg"))
                for camino in archivos:
                    n += 1
                    fr = cv2.imread(str(camino))
                    mask = np.load(self.masks() / f"mask_{n:05d}.npy")
                    if mask.shape[:2] != fr.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (fr.shape[1], fr.shape[0])) > 0
                    vis = _compone(mask, fr)
                    borde = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                    vis[(borde > 0) & (~mask)] = (0, 255, 255)
                    cv2.imwrite(str(tmp / f"{n:05d}.jpg"), vis)
                    self._prog("preview", n, len(archivos), f"Montando vista previa... {n}/{len(archivos)}")
                audio_orig = str(self.video) if self.video and self.video.exists() else None
                out = pre / "preview.mp4"
                if out.exists():
                    out.unlink()
                salida = tool.frames_a_video(str(tmp), str(out), self.fps, audio_path=audio_orig)
                self.cola.put(("hecho", True, f"Vista previa lista: {salida}"))
                self.cola.put(("preview_lista", True))
            except Exception as e:
                import traceback
                self.cola.put(("hecho", False, f"Error en vista previa:\n{traceback.format_exc()[-1200:]}"))

        threading.Thread(target=trabajo, daemon=True).start()

    def abrir_preview(self):
        salida = self._preview() / "preview.mp4"
        if salida.exists():
            os.startfile(str(salida))

    def exportar_ae(self):
        if self.ocupado or not self.meta.get("rastreado"):
            return
        ae = self._ae()
        shutil.rmtree(ae, ignore_errors=True)
        ae.mkdir(parents=True, exist_ok=True)
        nombre = _id_archivo(self.video.stem) if self.video else "sujeto"
        self.ocupado = True
        self._actualizar_estado()
        self._set_estado("Exportando máscaras para After Effects...")

        def trabajo():
            try:
                archivos = sorted(self.frames().glob("*.jpg"))
                n = 0
                for camino in archivos:
                    n += 1
                    fr = cv2.imread(str(camino), cv2.IMREAD_UNCHANGED)
                    mask = np.load(self.masks() / f"mask_{n:05d}.npy")
                    if mask.shape[:2] != fr.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (fr.shape[1], fr.shape[0])) > 0
                    if fr.shape[2] == 3:
                        bgra = cv2.cvtColor(fr, cv2.COLOR_BGR2BGRA)
                    else:
                        bgra = fr
                    bgra[:, :, 3] = (mask * 255).astype(np.uint8)
                    cv2.imwrite(str(ae / f"{nombre}_{n:04d}.png"), bgra)
                    self._prog("ae", n, len(archivos), f"Exportando PNG... {n}/{len(archivos)}")
                self.cola.put(("hecho", True,
                               f"Exportadas {n} máscaras a {ae}\nEn After Effects: File > Import, "
                               "selecciona el primer PNG y activa 'PNG Sequence' (alpha listo)."))
                self.cola.put(("preview_lista", False))
            except Exception as e:
                self.cola.put(("hecho", False, f"Error al exportar: {e}"))

        threading.Thread(target=trabajo, daemon=True).start()

    def limpiar_viejo(self):
        if self.ocupado or not self.clip_dir:
            return
        if not messagebox.askyesno("Limpiar carpetas antiguas",
                                   "Se borrarán máscaras, vista previa y exportaciones del clip actual.\n"
                                   "¿Continuar?"):
            return
        for sub in ("masks", "ae_export", "roto_tmp"):
            shutil.rmtree(self.clip_dir / sub, ignore_errors=True)
        shutil.rmtree(self._preview(), ignore_errors=True)
        salida_roto = self._ruta_roto()
        if salida_roto and salida_roto.exists():
            salida_roto.unlink()
        self.meta["rastreado"] = False
        self._guardar_meta()
        self._log("Carpetas de máscaras antiguas limpiadas.")
        self._set_estado("")
        self._actualizar_estado()

    def exportar_roto(self):
        if self.ocupado or not self.meta.get("rastreado"):
            return
        tmp = self.clip_dir / "roto_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        salida = self._ruta_roto()
        self.ocupado = True
        self._actualizar_estado()
        self._set_estado("Generando rotoscopia (fondo transparente)...")

        def trabajo():
            try:
                n = 0
                mascaras = {}
                for camino in sorted(self.frames().glob("*.jpg")):
                    n += 1
                    mascaras[n - 1] = np.load(self.masks() / f"mask_{n:05d}.npy")
                tool.exportar_rotoscopia(str(self.frames()), mascaras, str(tmp),
                                         softness=8, matting=False, progreso=self._prog)
                if salida.exists():
                    salida.unlink()
                final = tool.frames_a_video(str(tmp), str(salida), self.fps, con_alpha=True)
                self.cola.put(("hecho", True,
                               f"Rotoscopia lista SIN FONDO: {final}\n"
                               "Es un .mov ProRes 4444 con canal alpha (transparencia) "
                               "listo para After Effects / Premiere / DaVinci."))
                self.cola.put(("preview_lista", False))
            except Exception as e:
                import traceback
                self.cola.put(("hecho", False,
                               f"Error en rotoscopia:\n{traceback.format_exc()[-1200:]}"))

        threading.Thread(target=trabajo, daemon=True).start()

    def abrir_roto(self):
        ruta = self._ruta_roto()
        if ruta and ruta.exists():
            os.startfile(str(ruta))

    def abrir_resultados(self):
        if self.clip_dir and self.clip_dir.exists():
            os.startfile(str(self.clip_dir))

    # ---------------------------------------------------------- progreso / cola
    def _prog(self, etapa, actual, total, mensaje=""):
        pct = int(100 * actual / max(1, total))
        self.cola.put(("prog", pct, mensaje))

    def _poll_cola(self):
        try:
            while True:
                item = self.cola.get_nowait()
                if item[0] == "prog":
                    _, pct, msj = item
                    self.barra["value"] = pct
                    self.lbl_prog.config(text=msj)
                elif item[0] == "n_frames":
                    self.n_frames = item[1]
                elif item[0] == "pred_listo":
                    self._log(f"Modelo {item[1]} cargado.")
                    self.ocupado = False
                    self._actualizar_estado()
                    self._lanzar_rastreo(self.pred)
                elif item[0] == "preview_lista":
                    self._actualizar_estado()
                elif item[0] == "hecho":
                    _, ok, msj = item
                    self._log(msj)
                    self.ocupado = False
                    self.barra["value"] = 0
                    self.lbl_prog.config(text="")
                    if ok:
                        self._set_estado("Máscaras rastreadas listas para revisión y corrección."
                                         if "Máscaras listas" in msj else "")
                    self._actualizar_estado()
        except queue.Empty:
            pass
        self.root.after(120, self._poll_cola)


class SelWindow(tk.Toplevel):
    """Seleccionar (punto/caja) y corregir (puntos +/−) el sujeto en el frame 1."""

    def __init__(self, app, modo):
        super().__init__(app.root)
        self.app = app
        self.modo = modo
        self.title("Seleccionar sujeto en frame 1" if modo == "seleccionar"
                   else "Editar máscara seleccionada")
        self.geometry("980x660")
        self.minsize(720, 480)

        self.punto = app.meta.get("punto")
        self.box = app.meta.get("box")
        self.correcciones = list(app.meta.get("correcciones", []))

        self.herramienta = tk.StringVar(value="sujeto")
        self.radio_box = tk.BooleanVar(value=bool(app.meta.get("box")))
        self.pinting_box = False
        self.x0 = self.y0 = 0

        self._mas_vista = None
        self._seg_escala = 1.0
        self._seg_hilo = None
        self._seg_pend = False
        self._sig_seg = (None, None, ())

        if modo == "seleccionar":
            self.correcciones = [c for c in self.correcciones if False]
            self.mode_label = ttk.Label(
                self, text="Haz clic sobre el sujeto (verde) o arrastra una caja. "
                           "'Corregir +' y 'Corregir −' añaden puntos de refinado.",
                foreground="#606060", wraplength=940).pack(padx=10, pady=(8, 0))
        else:
            abrev = ("punto del sujeto de referencia" if app.meta.get("punto") is not None
                     else "caja de referencia")
            ttk.Label(self,
                      text=f"La máscara actual está visible. Añade correcciones: 'Corregir +' "
                           f"(lo que falta) o 'Corregir −' (lo que sobra), luego pulsa 'Guardar correcciones' "
                           f"y vuelve a 'Rastrear clip'.",
                      foreground="#606060", wraplength=940).pack(padx=10, pady=(8, 0))

        barra = ttk.Frame(self)
        barra.pack(fill="x", padx=10)
        ttk.Radiobutton(barra, text="Sujeto (punto)", variable=self.herramienta,
                        value="sujeto").pack(side="left")
        ttk.Radiobutton(barra, text="Caja (arrastrar)", variable=self.herramienta,
                        value="box").pack(side="left")
        ttk.Radiobutton(barra, text="Corregir +", variable=self.herramienta,
                        value="mas").pack(side="left", padx=(14, 0))
        ttk.Radiobutton(barra, text="Corregir −", variable=self.herramienta,
                        value="menos").pack(side="left")
        ttk.Button(barra, text="Deshacer último", command=self.deshacer).pack(side="right")
        ttk.Button(barra, text="Cancelar", command=self.destroy).pack(side="right", padx=4)

        self.canvas = tk.Canvas(self, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=6)
        self.lbl_seg = ttk.Label(self, text="", foreground="#1a5a8a")
        self.lbl_seg.pack(fill="x", padx=10)

        pie = ttk.Frame(self)
        pie.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(pie, text="Guardar", style="Accent.TButton",
                   command=self.guardar).pack(side="right")
        ttk.Button(pie, text="Guardar correcciones", command=self.guardar).pack(side="right", padx=4)

        self.canvas.bind("<Button-1>", self._clic)
        self.canvas.bind("<B1-Motion>", self._arrastrar)
        self.canvas.bind("<ButtonRelease-1>", self._soltar)
        self.canvas.bind("<Configure>", lambda e: self._redibujar())
        self.bind("<Escape>", lambda e: self.destroy())

        self._imagen_original = cv2.imread(app._frames_primer())
        h, w = self._imagen_original.shape[:2]
        self.tam = (w, h)
        self.scala = 1.0
        self._redibujar()
        self.after(150, self._segmentar_vista)

    def _img(self):
        fr = self._imagen_original
        m = None
        if self._mas_vista is not None:
            m = self._mas_vista
        elif self.app.meta.get("rastreado") and (self.app.masks() / "mask_00001.npy").exists():
            m = np.load(self.app.masks() / "mask_00001.npy")
        if m is None:
            return fr
        if m.shape[:2] != fr.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (fr.shape[1], fr.shape[0])) > 0
        return _compone(m, fr)

    def _segmentar_vista(self):
        if self.app.ocupado:
            self.lbl_seg.config(text="El clip está procesando; intenta de nuevo cuando termine.")
            return
        sig = (self.punto, self.box, tuple(self.correcciones))
        if self._seg_hilo is not None and self._seg_hilo.is_alive():
            self._seg_pend = True
            return
        if sig == self._sig_seg and self._mas_vista is not None:
            return
        self._sig_seg = sig
        if not (self.punto or self.box):
            self._mas_vista = None
            self._redibujar()
            self.lbl_seg.config(text="")
            return
        self._mas_vista = None
        self.lbl_seg.config(text="Segmentando para la vista previa...")
        self._seg_hilo = threading.Thread(
            target=self._seg_trabajo,
            args=(self.punto, self.box, list(self.correcciones)), daemon=True)
        self._seg_hilo.start()

    def _seg_trabajo(self, punto, box, correcciones):
        try:
            pred = self.app.pred
            if pred is None or self.app.pred_modelo != self.app.var_modelo.get():
                cp, cfg = self.app._ruta_modelo()
                if not cp.exists() or not cfg.exists():
                    self.after_idle(lambda: self.lbl_seg.config(
                        text="Modelo no encontrado. Descárgalo o cambia de modelo."))
                    return
                pred = tool.cargar_predictor(str(cp), str(cfg))
                self.app.pred = pred
                self.app.pred_modelo = self.app.var_modelo.get()
            fuente = self._imagen_original
            h, w = fuente.shape[:2]
            esc = min(1.0, 1280.0 / max(h, w))
            if esc < 1.0:
                trabajo = cv2.resize(fuente, (max(1, round(w * esc)), max(1, round(h * esc))))
            else:
                trabajo = fuente
            p_s = (punto[0] * esc, punto[1] * esc) if punto else None
            b_s = tuple(v * esc for v in box) if box else None
            e_s = [(x * esc, y * esc, et) for (x, y, et) in correcciones] or None
            m = tool.segmentar_imagen(pred, trabajo, punto_xy=p_s, box_xyxy=b_s,
                                      puntos_extra=e_s)
            self._seg_escala = esc

            def _fin():
                self._mas_vista = m
                self._redibujar()
                self.lbl_seg.config(
                    text="Vista previa en azul = lo que se exportará. "
                         "Usa 'Corregir +/−' para afinar y luego guarda.")
                if self._seg_pend:
                    self._seg_pend = False
                    self.after(50, self._segmentar_vista)

            self.after_idle(_fin)
        except Exception as e:
            self.after_idle(lambda: self.lbl_seg.config(
                text=f"Error de segmentación: {e}"))
        finally:
            self._seg_hilo = None

    def _redibujar(self):
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        self.scala = min(cw / self.tam[0], ch / self.tam[1])
        nuevo = (max(1, int(self.tam[0] * self.scala)), max(1, int(self.tam[1] * self.scala)))
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(self._img(), cv2.COLOR_BGR2RGB))
        img = img.resize(nuevo)
        try:
            self._tkimg = ImageTk.PhotoImage(img)
        except Exception:
            self._tkimg = None
        self.canvas.delete("all")
        if self._tkimg:
            self.canvas.create_image(0, 0, image=self._tkimg, anchor="nw")
        if self.radio_box.get() and self.box:
            self.canvas.create_rectangle(*self._pix(self.box), outline="#00ff88",
                                         width=2, tags="sel")
        elif self.punto:
            x, y = int(self.punto[0] * self.scala), int(self.punto[1] * self.scala)
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#00ff88",
                                    outline="white", tags="sel")
        for x, y, et in self.correcciones:
            px, py = int(x * self.scala), int(y * self.scala)
            color = "#00ff88" if et else "#ff4444"
            self.canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill=color,
                                    outline="white", tags="corr")

    def _pix(self, xy):
        return (int(xy[0] * self.scala), int(xy[1] * self.scala),
                int(xy[2] * self.scala), int(xy[3] * self.scala))

    def _coord(self, ev):
        return max(0, min(self.tam[0] - 1, int(ev.x / self.scala))), \
               max(0, min(self.tam[1] - 1, int(ev.y / self.scala)))

    def _clic(self, ev):
        x, y = self._coord(ev)
        if self.herramienta.get() == "sujeto":
            self.punto = (x, y)
            self.box = None
            self.radio_box.set(False)
            self._redibujar()
            self._segmentar_vista()
        elif self.herramienta.get() == "box":
            self.pinting_box = True
            self.x0, self.y0 = x, y
            self.punto = None
            self.radio_box.set(True)
        elif self.herramienta.get() == "mas":
            self.correcciones.append((x, y, 1))
            self._redibujar()
            self._segmentar_vista()
        elif self.herramienta.get() == "menos":
            self.correcciones.append((x, y, 0))
            self._redibujar()
            self._segmentar_vista()

    def _arrastrar(self, ev):
        if self.pinting_box:
            x, y = self._coord(ev)
            self.canvas.delete("boxprev")
            self.canvas.create_rectangle(self.x0 * self.scala, self.y0 * self.scala,
                                         x * self.scala, y * self.scala,
                                         outline="#00ff88", width=2, tags="boxprev")

    def _soltar(self, ev):
        if self.pinting_box:
            self.pinting_box = False
            x, y = self._coord(ev)
            xa, ya = self.x0, self.y0
            x1, y1 = min(xa, x), min(ya, y)
            x2, y2 = max(xa, x), max(ya, y)
            if x2 - x1 < 4 or y2 - y1 < 4:
                self.radio_box.set(False)
                self.punto = ((xa + x) // 2, (ya + y) // 2)
                self.box = None
            else:
                self.box = (x1, y1, x2, y2)
            self._redibujar()
            self._segmentar_vista()

    def deshacer(self):
        if self.correcciones:
            self.correcciones.pop()
            self._redibujar()
            self._segmentar_vista()

    def guardar(self):
        app = self.app
        if not (self.punto or self.box):
            messagebox.showwarning("Falta la selección",
                                   "Haz clic sobre el sujeto o arrastra una caja.")
            return
        app.meta["punto"] = list(self.punto) if self.punto else None
        app.meta["box"] = list(self.box) if self.box else None
        app.meta["correcciones"] = self.correcciones
        app._guardar_meta()
        app._log(("Selección del sujeto guardada."
                  if self.modo == "seleccionar" else
                  "Correcciones guardadas. Vuelve a 'Rastrear clip' para aplicar."))
        app._actualizar_estado()
        self.destroy()


class ReviewWindow(tk.Toplevel):
    """Revisar y corregir frames: pintar +/− sobre la máscara por frame."""

    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("Revisar y corregir frames")
        self.geometry("980x720")
        self.minsize(720, 520)

        self.archivos = sorted(app.frames().glob("*.jpg"))
        self.n_frames = len(self.archivos)
        self.idx = 0
        self.mascara = None
        self.base = None
        self.pincel = tk.StringVar(value="mas")
        self.radio = tk.IntVar(value=1)
        tam_radio = tk.DoubleVar(value=30)

        barra = ttk.Frame(self)
        barra.pack(fill="x", padx=10, pady=(8, 0))
        self.slider = tk.Scale(barra, from_=1, to=self.n_frames, orient="horizontal",
                               command=lambda v: self.cargar(int(v) - 1), length=420)
        self.slider.set(1)
        self.slider.pack(side="left")
        self.lbl_frame = ttk.Label(barra, text="Frame 1")
        self.lbl_frame.pack(side="left", padx=8)

        ttk.Radiobutton(barra, text="Añadir zona (pintar +)", variable=self.radio,
                        value=1).pack(side="left", padx=(16, 0))
        ttk.Radiobutton(barra, text="Quitar zona (pintar −)", variable=self.radio,
                        value=0).pack(side="left")
        ttk.Label(barra, text="Tamaño:").pack(side="left", padx=(16, 0))
        self.brus = ttk.Scale(barra, from_=4, to=90, orient="horizontal",
                              variable=tam_radio)
        self.brus.pack(side="left")
        ttk.Button(barra, text="Restaurar frame original",
                   command=self.restaurar).pack(side="right")

        self.canvas = tk.Canvas(self, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=6)
        self.canvas.bind("<B1-Motion>", self.pintar)
        self.canvas.bind("<Button-1>", self.pintar)
        self.canvas.bind("<Configure>", lambda e: self._redibujar())

        pie = ttk.Frame(self)
        pie.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(pie, text="Las correcciones se guardan automáticamente al cambiar de frame.",
                  foreground="#707070").pack(side="left")
        ttk.Button(pie, text="Guardar y cerrar", style="Accent.TButton",
                   command=self.cerrar).pack(side="right")

        self.escala = 1.0
        self.cargar(0)

    def _imagen(self):
        fr = cv2.imread(str(self.archivos[self.idx]))
        if self.mascara is None:
            return fr
        return _compone(self.mascara, fr)

    def cargar(self, idx):
        if not (0 <= idx < self.n_frames):
            return
        if self.base is not None:
            np.save(self.app.masks() / f"mask_{self.idx + 1:05d}.npy", self.mascara)
        self.idx = idx
        self.lbl_frame.config(text=f"Frame {idx + 1}")
        self.base = np.load(self.app.masks() / f"mask_{idx + 1:05d}.npy")
        self.mascara = self.base.copy()
        fr = cv2.imread(str(self.archivos[idx]))
        self.tam = (fr.shape[1], fr.shape[0])
        self._redibujar()

    def _redibujar(self):
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        self.escala = min(cw / self.tam[0], ch / self.tam[1])
        nuevo = (max(1, int(self.tam[0] * self.escala)), max(1, int(self.tam[1] * self.escala)))
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(self._imagen(), cv2.COLOR_BGR2RGB)).resize(nuevo)
        try:
            self._tkimg = ImageTk.PhotoImage(img)
        except Exception:
            self._tkimg = None
        self.canvas.delete("all")
        if self._tkimg:
            self.canvas.create_image(0, 0, image=self._tkimg, anchor="nw")

    def pintar(self, ev):
        x = max(0, min(self.tam[0] - 1, int(ev.x / self.escala)))
        y = max(0, min(self.tam[1] - 1, int(ev.y / self.escala)))
        r = max(2, int(float(self.brus.get()) / 2))
        valor = 1 if self.radio.get() else 0
        m = self.mascara.astype(np.uint8)
        cv2.circle(m, (x, y), r, valor, -1)
        self.mascara = m > 0
        self._redibujar()

    def restaurar(self):
        self.mascara = self.base.copy()
        self._redibujar()

    def cerrar(self):
        if self.base is not None:
            np.save(self.app.masks() / f"mask_{self.idx + 1:05d}.npy", self.mascara)
        self.app._log("Revisión y corrección de frames guardada.")
        self.destroy()


def main():
    root = tk.Tk()
    try:
        estilo = ttk.Style(root)
        estilo.theme_use("clam")
        estilo.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    except Exception:
        pass
    RotoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
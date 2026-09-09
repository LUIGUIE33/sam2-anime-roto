"""
sam2_gui.py
--------------------------------------------------------------------
Interfaz gráfica para sam2_video_tool.py:
  - Eliges un video, ves su primer frame y haces clic sobre el objeto
    (o arrastras para dibujar una caja).
  - SAM 2 rastrea el objeto en todo el video.
  - Exportas en modo "autocrop" (vertical 9:16 siguiendo al objeto)
    o "rotoscopia" (fondo transparente con alpha).

Uso:
    python sam2_gui.py
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import cv2
from PIL import Image, ImageTk

import sam2_video_tool as tool

ANCHO_PREVIEW = 800
ALTO_PREVIEW = 300


class SAM2App:
    def __init__(self, root):
        self.root = root
        root.title("SAM 2 - Segmenta y recorta tus videos")

        ancho_pantalla = root.winfo_screenwidth()
        alto_pantalla = root.winfo_screenheight()
        ancho_v = min(960, ancho_pantalla - 40)
        alto_v = min(700, alto_pantalla - 80)
        root.geometry(f"{ancho_v}x{alto_v}")
        root.minsize(880, 560)

        self.ruta_video = tk.StringVar()
        self.ruta_checkpoint = tk.StringVar(
            value=str(Path(__file__).parent / "sam2-main" / "checkpoints" / "sam2.1_hiera_tiny.pt"))
        self.ruta_config = tk.StringVar(
            value=str(Path(__file__).parent / "sam2-main" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"))
        self.ruta_salida = tk.StringVar()
        self.modo = tk.StringVar(value="autocrop")
        self.ancho_out = tk.StringVar(value="1080")
        self.alto_out = tk.StringVar(value="1920")
        self.selector = tk.StringVar(value="punto")
        self.matting = tk.BooleanVar(value=False)
        self.softness = tk.DoubleVar(value=8)
        self.modelo = tk.StringVar(value="Rápido (Tiny)")
        self.modelo_desc = tk.StringVar(value="")
        self.MODELOS = {
            "Rápido (Tiny)": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml",
                              "Rápido. Menos finura en bordes finos."),
            "Equilibrado (Base+)": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml",
                                    "Balance equilibrado entre velocidad y precisión."),
            "Calidad (Large)": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml",
                                "Máxima precisión. Requiere ~4 GB de VRAM."),
        }

        self.frame_original = None     # primer frame (BGR), original
        self.h_orig = self.w_orig = 0
        self.escala = 1.0             # tamano original / tamano mostrado
        self.punto = None             # (x, y) en coords originales
        self.box = None               # (x1, y1, x2, y2) en coords originales

        self.punto_item = None
        self.box_item = None
        self.inicio_box = None

        self.cola = queue.Queue()
        self.procesando = False
        self.progress = None
        self._widgets_controles = []

        self._build_widgets()
        self.root.after(150, self._poll_cola)

    # ------------------------------------------------------------------
    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        # ---- Video de entrada ----
        ttk.Label(frm, text="Video de entrada:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.ruta_video).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Examinar...", command=self.seleccionar_video).grid(row=0, column=2, **pad)

        # ---- Preview + canvas ----
        marco = ttk.LabelFrame(frm, text="Primer frame - haz clic sobre el objeto (o arrastra para caja)", padding=4)
        marco.grid(row=1, column=0, columnspan=3, sticky="nsew", **pad)
        self.canvas = tk.Canvas(marco, width=ANCHO_PREVIEW, height=ALTO_PREVIEW, bg="#1e1e1e",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.lbl_info = ttk.Label(marco, text="Selecciona un video para ver el primer frame.",
                                  foreground="#888888")
        self.lbl_info.pack(fill="x", padx=4)

        # ---- Opciones ----
        frm_opc = ttk.LabelFrame(frm, text="Objeto y modo", padding=8)
        frm_opc.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Radiobutton(frm_opc, text="Clic (punto)", variable=self.selector, value="punto").grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frm_opc, text="Caja (arrastrar)", variable=self.selector, value="box").grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(frm_opc, text="Limpiar selección", command=self.limpiar_seleccion).grid(row=0, column=2, **pad)

        ttk.Separator(frm_opc, orient="vertical").grid(row=0, column=3, sticky="ns", padx=8)

        ttk.Radiobutton(frm_opc, text="autocrop (recorte 9:16 que sigue al objeto)",
                        variable=self.modo, value="autocrop",
                        command=self._actualizar_salto_sugerido).grid(row=0, column=4, sticky="w", **pad)
        ttk.Radiobutton(frm_opc, text="rotoscopia (fondo transparente con alpha)",
                        variable=self.modo, value="rotoscopia").grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(frm_opc, text="Salida vertical:").grid(row=1, column=4, sticky="e", **pad)
        frm_dim = ttk.Frame(frm_opc)
        frm_dim.grid(row=1, column=5, sticky="w", **pad)
        ttk.Entry(frm_dim, textvariable=self.ancho_out, width=6).pack(side="left")
        ttk.Label(frm_dim, text="x").pack(side="left", padx=2)
        ttk.Entry(frm_dim, textvariable=self.alto_out, width=6).pack(side="left")

        # ---- Modelo ----
        frm_modelo = ttk.LabelFrame(frm, text="Modelo SAM 2 (checkpoint y config)", padding=8)
        frm_modelo.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(frm_modelo, text="Modelo:").grid(row=0, column=0, sticky="w", **pad)
        self.cb_modelo = ttk.Combobox(frm_modelo, textvariable=self.modelo,
                                      values=list(self.MODELOS.keys()) + ["Personalizado..."],
                                      state="readonly", width=24)
        self.cb_modelo.grid(row=0, column=1, sticky="ew", **pad)
        self.cb_modelo.bind("<<ComboboxSelected>>", lambda e: self._modelo_cmd())
        self.lbl_modelo_desc = ttk.Label(frm_modelo, textvariable=self.modelo_desc,
                                         foreground="#707070")
        self.lbl_modelo_desc.grid(row=0, column=2, columnspan=2, sticky="w", **pad)
        ttk.Label(frm_modelo, text="Checkpoint:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_modelo, textvariable=self.ruta_checkpoint).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm_modelo, text="Examinar...",
                   command=lambda: self._browse(self.ruta_checkpoint)).grid(row=1, column=2, **pad)
        ttk.Label(frm_modelo, text="Config:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm_modelo, textvariable=self.ruta_config).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frm_modelo, text="Examinar...",
                   command=lambda: self._browse(self.ruta_config)).grid(row=2, column=2, **pad)
        frm_modelo.columnconfigure(1, weight=1)
        self._modelo_cmd()

        # ---- Edge Quality ----
        frm_eq = ttk.LabelFrame(frm, text="Edge Quality", padding=8)
        frm_eq.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Checkbutton(frm_eq, text="Refine edges with matting (slower, cleaner result)",
                        variable=self.matting).grid(row=0, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(frm_eq, text="Edge softness (px):").grid(row=1, column=0, sticky="w", **pad)
        self.scale_softness = ttk.Scale(frm_eq, from_=0, to=15, variable=self.softness,
                                        command=self._soft_cmd)
        self.scale_softness.grid(row=1, column=1, sticky="ew", **pad)
        self.lbl_soft = ttk.Label(frm_eq, text="8", width=3)
        self.lbl_soft.grid(row=1, column=2, sticky="w", **pad)
        ttk.Label(frm_eq, foreground="#707070",
                  text="Edge Quality settings apply during tracking and export. "
                       "Enabling matting increases processing time per frame.").grid(
            row=2, column=0, columnspan=3, sticky="w", **pad)

        # ---- Salida ----
        ttk.Label(frm, text="Guardar como:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.ruta_salida).grid(row=5, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Examinar...", command=self._seleccionar_salida).grid(row=5, column=2, **pad)

        # ---- Progreso ----
        ttk.Label(frm, text="Estado:").grid(row=6, column=0, sticky="w", **pad)
        self.progress = ttk.Progressbar(frm, maximum=100.0, value=0)
        self.progress.grid(row=6, column=1, columnspan=2, sticky="ew", **pad)

        self.txt_log = tk.Text(frm, height=6, state="disabled", bg="#1e1e1e", fg="#c8c8c8",
                               font=("Consolas", 9))
        self.txt_log.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)

        # ---- Acciones ----
        frm_acc = ttk.Frame(frm)
        frm_acc.grid(row=8, column=0, columnspan=3, sticky="e", **pad)
        btn = ttk.Button(frm_acc, text="Procesar video", command=self.procesar,
                         style="Accent.TButton")
        btn.pack(side="right", padx=4)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)

        self.chk_matting = frm_eq.winfo_children()[0]
        self.btn_procesar = btn
        self._widgets_controles = [self.chk_matting, self.btn_procesar]

    def _soft_cmd(self, val):
        self.lbl_soft.config(text=str(int(round(float(val)))))

    # ------------------------------------------------------------------
    def seleccionar_video(self):
        ruta = filedialog.askopenfilename(
            title="Elige tu video",
            filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi *.webm *.m2ts"), ("Todos", "*.*")])
        if not ruta:
            return
        self.ruta_video.set(ruta)
        Path(ruta)
        nombre = Path(ruta).stem
        if self.modo.get() == "autocrop":
            self.ruta_salida.set(str(Path(ruta).with_name(f"{nombre}_vertical.mp4")))
        else:
            self.ruta_salida.set(str(Path(ruta).with_name(f"{nombre}_alpha.mov")))
        self._mostrar_frame(ruta)

    def _mostrar_frame(self, ruta_video):
        cap = cv2.VideoCapture(ruta_video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            self.canvas.delete("all")
            self.lbl_info.config(text="No se pudo leer el primer frame de este video.")
            self.frame_original = None
            return

        self.frame_original = frame
        self.h_orig, self.w_orig = frame.shape[:2]
        escala = min(ANCHO_PREVIEW / self.w_orig, ALTO_PREVIEW / self.h_orig, 1.0)
        self.escala = 1.0 / escala
        disp_w, disp_h = int(self.w_orig * escala), int(self.h_orig * escala)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        ox, oy = (ANCHO_PREVIEW - disp_w) // 2, (ALTO_PREVIEW - disp_h) // 2
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
        self._offsets = (ox, oy)

        self.lbl_info.config(
            text=f"{self.w_orig}x{self.h_orig} px  |  clic = punto    arrastrar = caja")
        self.punto = None
        self.box = None
        self.punto_item = self.box_item = None

    # ------------------------------------------------------------------
    def _a_originales(self, c_x, c_y):
        ox, oy = self._offsets
        x = int(round((c_x - ox) * self.escala))
        y = int(round((c_y - oy) * self.escala))
        x = max(0, min(x, self.w_orig - 1))
        y = max(0, min(y, self.h_orig - 1))
        return x, y

    def _on_click(self, event):
        if self.frame_original is None:
            return
        if self.selector.get() == "box":
            self.inicio_box = (event.x, event.y)
            self._clear_overlay()
            self.box_item = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                         outline="#ff3b30", width=2)
            return
        x, y = self._a_originales(event.x, event.y)
        self.punto = (x, y)
        self.limpiar_seleccion(solo_overlay=True)
        self.punto_item = self.canvas.create_oval(event.x - 6, event.y - 6, event.x + 6, event.y + 6,
                                                  outline="#4cd964", width=2)
        self.box = None
        self.lbl_info.config(text=f"Clic en ({x}, {y})")

    def _on_drag(self, event):
        if self.inicio_box is None:
            return
        self.canvas.coords(self.box_item, self.inicio_box[0], self.inicio_box[1], event.x, event.y)

    def _on_release(self, event):
        if self.inicio_box is None:
            return
        x1, y1 = self._a_originales(self.inicio_box[0], self.inicio_box[1])
        x2, y2 = self._a_originales(event.x, event.y)
        self.inicio_box = None
        if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
            self._clear_overlay()
            self.box = None
            return
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.box = (x1, y1, x2, y2)
        self.punto = None
        self.lbl_info.config(text=f"Caja: ({x1}, {y1}) -> ({x2}, {y2})")

    def _clear_overlay(self):
        if self.punto_item:
            self.canvas.delete(self.punto_item)
            self.punto_item = None
        if self.box_item:
            self.canvas.delete(self.box_item)
            self.box_item = None

    def limpiar_seleccion(self, solo_overlay=False):
        self._clear_overlay()
        if not solo_overlay:
            self.punto = None
            self.box = None
            self.lbl_info.config(text="Selección borrada.")

    def _actualizar_salto_sugerido(self):
        if self.ruta_video.get():
            ruta = Path(self.ruta_video.get())
            self.ruta_salida.set(str(ruta.with_name(f"{ruta.stem}_vertical.mp4")))

    def _seleccionar_salida(self):
        if self.modo.get() == "rotoscopia":
            tipos = [("Video con alpha (MOV)", "*.mov"), ("Video con alpha (WebM)", "*.webm"), ("Todos", "*.*")]
            defext = ".mov"
        else:
            tipos = [("Video MP4", "*.mp4"), ("Todos", "*.*")]
            defext = ".mp4"
        ruta = filedialog.asksaveasfilename(title="Guardar resultado como",
                                            defaultextension=defext, filetypes=tipos)
        if ruta:
            self.ruta_salida.set(ruta)

    def _browse(self, var):
        ruta = filedialog.askopenfilename(title="Elegir archivo")
        if ruta:
            var.set(ruta)

    def _modelo_cmd(self):
        datos = self.MODELOS.get(self.modelo.get())
        if not datos:
            self.modelo_desc.set("Usa las rutas manuales de checkpoint y config.")
            return
        base = Path(__file__).parent / "sam2-main" / "sam2"
        self.ruta_checkpoint.set(str(Path(__file__).parent / "sam2-main" / "checkpoints" / datos[0]))
        self.ruta_config.set(str(base / datos[1]))
        self.modelo_desc.set(datos[2])

    # ------------------------------------------------------------------
    def procesar(self):
        if self.procesando:
            return
        video = self.ruta_video.get().strip()
        cp = self.ruta_checkpoint.get().strip()
        cfg = self.ruta_config.get().strip()
        salida = self.ruta_salida.get().strip()

        errores = []
        if not video or not Path(video).exists():
            errores.append("- El video de entrada no existe.")
        if not cp or not Path(cp).exists():
            errores.append("- El checkpoint no existe.")
        if not cfg or not Path(cfg).exists():
            errores.append("- El archivo de config no existe.")
        if not self.punto and not self.box:
            errores.append("- Haz clic sobre el objeto (o arrastra una caja) en el primer frame.")
        if not salida:
            errores.append("- Indica un nombre de archivo de salida.")
        if self.modo.get() == "autocrop":
            try:
                ancho = int(self.ancho_out.get())
                alto = int(self.alto_out.get())
            except ValueError:
                errores.append("- Ancho/alto de autocrop deben ser números.")
                ancho = alto = None
        if errores:
            messagebox.showerror("Faltan datos", "\n".join(errores))
            return

        self.procesando = True
        self.progress["value"] = 0
        self._log(f"\n=== Procesando: {Path(video).name} ===\n")
        self.root.config(cursor="watch")
        self._toggle_ui(False)

        hilo = threading.Thread(target=self._trabajo,
                                args=(video, cp, cfg, salida), daemon=True)
        hilo.start()

    def _toggle_ui(self, activo):
        estado = "normal" if activo else "disabled"
        for w in (self.scale_softness,):
            try:
                w.config(state=estado)
            except Exception:
                pass
        for child in self._widgets_controles:
            try:
                child.config(state=estado)
            except Exception:
                pass

    def _trabajo(self, video, cp, cfg, salida):
        tmp = Path("sam2_trabajo_tmp")
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            carpeta_frames = tmp / "frames"
            carpeta_proc = tmp / "procesados"

            self._info("extraer", 0, 1, "Extrayendo frames...")
            fps = tool.extraer_frames(video, str(carpeta_frames), progreso=self._progreso)

            tiene_audio = tool.obtener_audio(video, str(tmp / "audio.aac"))

            self._info("cargar", 0, 1, "Cargando SAM 2...")
            predictor = tool.cargar_predictor(cp, cfg)

            mascaras = tool.segmentar_video(
                predictor, str(carpeta_frames),
                punto_xy=tuple(self.punto) if self.punto else None,
                box_xyxy=tuple(self.box) if self.box else None,
                progreso=self._progreso)

            modo = self.modo.get()
            softness = int(round(float(self.softness.get())))
            matting = bool(self.matting.get())
            if matting:
                self.cola.put(("log", f"Edge Quality: matting por trimap activado (softness={softness}px)"))
            if modo == "rotoscopia":
                tool.exportar_rotoscopia(str(carpeta_frames), mascaras, str(carpeta_proc),
                                         softness=softness, matting=matting,
                                         progreso=self._progreso)
                salida = tool.frames_a_video(str(carpeta_proc), salida, fps,
                                             audio_path=str(tmp / "audio.aac") if tiene_audio else None,
                                             con_alpha=True)
            else:
                ancho = int(self.ancho_out.get())
                alto = int(self.alto_out.get())
                tool.exportar_autocrop(str(carpeta_frames), mascaras, str(carpeta_proc),
                                       ancho_salida=ancho, alto_salida=alto,
                                       softness=softness, matting=matting,
                                       progreso=self._progreso)
                tool.frames_a_video(str(carpeta_proc), salida, fps,
                                    audio_path=str(tmp / "audio.aac") if tiene_audio else None,
                                    con_alpha=False)

            self.cola.put(("done", True, f"\nListo -> {salida}"))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.cola.put(("done", False, f"\nERROR: {e}\n\n{tb}"))
        finally:
            if tmp.exists():
                try:
                    shutil.rmtree(tmp)
                except Exception:
                    pass

    def _progreso(self, etapa, actual, total, mensaje):
        if total and total > 0:
            frac = actual / total
        else:
            frac = 0.0
        self.cola.put(("progress", frac, mensaje))

    def _info(self, etapa, actual, total, mensaje):
        self.cola.put(("progress", 0.0, mensaje))

    # ------------------------------------------------------------------
    def _poll_cola(self):
        try:
            while True:
                item = self.cola.get_nowait()
                if item[0] == "progress":
                    _, frac, msg = item
                    self.progress["value"] = frac * 100
                elif item[0] == "log":
                    self._log(item[1])
                elif item[0] == "done":
                    _, ok, msg = item
                    self._log(msg)
                    self.procesando = False
                    self.root.config(cursor="")
                    self._toggle_ui(True)
                    if ok:
                        self.progress["value"] = 100
                        messagebox.showinfo("Terminado", msg)
                    else:
                        messagebox.showerror("Error", msg[:1500])
        except queue.Empty:
            pass
        self.root.after(150, self._poll_cola)

    def _log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    except Exception:
        pass
    SAM2App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
"""
sam2_video_tool.py
--------------------------------------------------------------------
Herramienta para usar SAM 2 (Segment Anything 2, de Meta) y recortar
/aislar un objeto de un video automáticamente.

Flujo:
  1. Extrae los frames del video con ffmpeg.
  2. Recibe un punto (clic) o un box sobre el objeto a segmentar.
  3. SAM 2 propaga la máscara de ese objeto a lo largo de todo el
     video (tracking automático).
  4. Exporta el resultado en dos modos:
       - "rotoscopia": video con fondo transparente (ProRes 4444 o
         WebM VP9 con alpha), solo el objeto segmentado.
       - "autocrop": recorte que sigue al objeto por el centro de su
         máscara (útil para verticalizar clips a 9:16).

Las funciones aceptan un callback `progreso(etapa, actual, total,
mensaje)` para mostrar avance desde una GUI.

Requiere GPU NVIDIA con CUDA para una velocidad razonable.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def extraer_frames(video_path: str, carpeta_frames: str, fps: float = None,
                   progreso=None) -> float:
    """Extrae todos los frames del video a JPGs numerados. Devuelve el fps real."""
    os.makedirs(carpeta_frames, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps_real = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if progreso:
        progreso("extraer", 0, 1, "Extrayendo frames con ffmpeg...")
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [os.path.join(carpeta_frames, "%05d.jpg")]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if progreso:
        progreso("extraer", 1, 1, "Frames extraídos.")
    return fps if fps else fps_real


def obtener_audio(video_path: str, salida_audio: str) -> bool:
    """Extrae el audio original para volver a pegarlo al final."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", salida_audio]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def frames_a_video(carpeta_frames: str, salida: str, fps: float,
                   audio_path: str = None, con_alpha: bool = False):
    """Junta los frames procesados en un video final, opcionalmente con audio."""
    patron = os.path.join(carpeta_frames, "%05d.png" if con_alpha else "%05d.jpg")

    ext = Path(salida).suffix.lower()

    if con_alpha:
        if ext == ".webm":
            # WebM/VP9 admite transparencia (alpha plana)
            codec_args = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                          "-b:v", "0", "-crf", "29", "-row-mt", "1", "-auto-alt-ref", "0"]
        else:
            if ext != ".mov":
                salida = str(Path(salida).with_suffix(".mov"))
            # ProRes 4444 conserva transparencia. SOLO va en contenedor MOV;
            # MP4 no soporta alpha y ffmpeg fracasa con "invalid argument".
            codec_args = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
    else:
        codec_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]

    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", patron]

    if audio_path and os.path.exists(audio_path):
        cmd += ["-i", audio_path, "-c:a", "aac", "-shortest"]

    cmd += codec_args + [salida]
    subprocess.run(cmd, check=True)
    return salida


def cargar_predictor(checkpoint: str, config: str):
    """Carga el modelo SAM 2 para video."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    import sam2

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        print("[AVISO] No se detectó GPU. Va a correr en CPU y será MUY lento.")
        device = "cpu"

    if os.path.isfile(config):
        paquete = Path(sam2.__file__).parent
        try:
            config = str(Path(config).resolve().relative_to(paquete)).replace("\\", "/")
        except ValueError:
            pass

    predictor = build_sam2_video_predictor(config, checkpoint, device=device)
    return predictor


def segmentar_video(predictor, carpeta_frames: str, punto_xy=None, box_xyxy=None,
                    frame_idx_inicial: int = 0, puntos_extra=None, progreso=None):
    """
    Corre la segmentación + tracking de SAM 2 sobre todos los frames.
    Devuelve un dict {frame_idx: mascara_booleana (H, W)}.

    punto_xy: tupla (x, y) de un clic positivo sobre el objeto en el primer frame.
    box_xyxy: opcional, (x1, y1, x2, y2) si se prefiere un box en vez de un punto.
    puntos_extra: opcional, lista de (x, y, etiqueta) con etiqueta 1 (añadir)
                  o 0 (quitar), aplicados en el frame inicial como correcciones.
    """
    import torch

    mascaras_por_frame = {}

    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=carpeta_frames)

        object_id = 1

        if box_xyxy is not None:
            box = np.array(box_xyxy, dtype=np.float32)
            _, _, _ = predictor.add_new_points_or_box(
                state, frame_idx=frame_idx_inicial, obj_id=object_id, box=box
            )
        else:
            points = np.array([[punto_xy[0], punto_xy[1]]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            _, _, _ = predictor.add_new_points_or_box(
                state, frame_idx=frame_idx_inicial, obj_id=object_id,
                points=points, labels=labels,
            )

        if puntos_extra:
            por_etiqueta = {}
            for x, y, et in puntos_extra:
                por_etiqueta.setdefault(int(et), []).append((float(x), float(y)))
            for etiqueta, pts in por_etiqueta.items():
                pp = np.array(pts, dtype=np.float32)
                ll = np.array([etiqueta] * len(pts), dtype=np.int32)
                _, _, _ = predictor.add_new_points_or_box(
                    state, frame_idx=frame_idx_inicial, obj_id=object_id,
                    points=pp, labels=ll,
                )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            mascara = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            mascaras_por_frame[out_frame_idx] = mascara
            if progreso:
                progreso("segmentar", out_frame_idx, len(mascaras_por_frame),
                         f"Rastreando objeto... frame {out_frame_idx}")

    return mascaras_por_frame


def segmentar_imagen(predictor, imagen_bgr, punto_xy=None, box_xyxy=None,
                     puntos_extra=None):
    """Segmenta UNA imagen (no video). Devuelve la máscara booleana (H, W)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cv2.imwrite(os.path.join(td, "00000.jpg"), imagen_bgr)
        mascaras = segmentar_video(predictor, td, punto_xy, box_xyxy,
                                   puntos_extra=puntos_extra, progreso=None)
    return mascaras[0]


def limpiar_mascara(mascara, kernel=3):
    """Punto 2 (siempre activo): quita islas sueltas y huecos con OPEN+CLOSE."""
    m = mascara.astype(np.uint8)
    k = np.ones((kernel, kernel), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def _alpha_antialias(mascara, sigma=1.0):
    """Alpha con borde suave (antialiasing) en vez de binario cortado a sierra."""
    return cv2.GaussianBlur(mascara.astype(np.float32), (0, 0), sigmaX=sigma)


def _matte_por_trimap(frame, mascara, softness=8):
    """
    Punto 3: genera un trimap (sujeto / banda desconocida / fondo) y lo
    resuelve con matting real (pymatting, Large Kernel Matting).
    El ancho de la banda desconocida = softness (px).
    Trabaja a resolución reducida (lado max ~1000 px) y devuelve el alpha
    a resolución nativa.
    """
    import pymatting

    h, w = mascara.shape[:2]
    max_lado = 1000.0
    esc = min(1.0, max_lado / max(h, w))
    tam = (max(1, int(round(w * esc))), max(1, int(round(h * esc)))) if esc < 1.0 else (w, h)

    radius = max(1, int(round(softness)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    fg = cv2.erode(mascara.astype(np.uint8), k)
    bg = cv2.dilate(mascara.astype(np.uint8), k)

    trimap = np.full((h, w), 128, dtype=np.uint8)
    trimap[fg > 0] = 255
    trimap[bg == 0] = 0

    if tam != (w, h):
        img = cv2.resize(frame, tam, interpolation=cv2.INTER_LINEAR)
        tri = cv2.resize(trimap, tam, interpolation=cv2.INTER_NEAREST)
    else:
        img, tri = frame, trimap

    alpha = pymatting.estimate_alpha_lkm(
        img.astype(np.float64) / 255.0,
        tri.astype(np.float64) / 255.0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    if tam != (w, h):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return alpha


def refinar_mascara(frame, mascara, softness=8, matting=False):
    """
    Aplica el refinado de bordes a una máscara:
      - siempre: limpieza morfológica (OPEN+CLOSE).
      - matting=True: alpha real vía trimap (banda = softness). Si el motor
        no está disponible, cae a alpha suavizada.
      - matting=False: alpha con antialias gaussiano.
    Devuelve (alpha_float_0_1, mascara_limpia_bool).
    """
    limpia = limpiar_mascara(mascara)
    if matting:
        try:
            alpha = _matte_por_trimap(frame, limpia, softness)
        except Exception:
            alpha = _alpha_antialias(limpia)
    else:
        alpha = _alpha_antialias(limpia)
    return alpha, limpia


def exportar_rotoscopia(carpeta_frames: str, mascaras: dict, carpeta_salida: str,
                        softness: int = 8, matting: bool = False, progreso=None):
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = sorted(Path(carpeta_frames).glob("*.jpg"))
    total = len(archivos)

    for i, archivo in enumerate(archivos):
        frame = cv2.imread(str(archivo))
        h, w = frame.shape[:2]
        mascara = mascaras.get(i)

        if mascara is None:
            alpha = np.zeros((h, w), dtype=np.uint8)
        else:
            alpha, _ = refinar_mascara(frame, mascara, softness=softness, matting=matting)
            alpha = (alpha * 255.0).astype(np.uint8)

        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        cv2.imwrite(str(Path(carpeta_salida) / f"{i+1:05d}.png"), bgra)
        if progreso:
            progreso("rotoscopia", i + 1, total, f"Generando fondo transparente... {i+1}/{total}")


def exportar_autocrop(carpeta_frames: str, mascaras: dict, carpeta_salida: str,
                      ancho_salida: int = 1080, alto_salida: int = 1920,
                      suavizado: int = 15, softness: int = 8, matting: bool = False,
                      progreso=None):
    """
    Recorta cada frame centrando el crop en el centroide del objeto
    detectado, con suavizado para que la cámara no "tiemble".
    Las máscaras se limpian morfológicamente (OPEN+CLOSE) antes de
    calcular el centroide.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    archivos = sorted(Path(carpeta_frames).glob("*.jpg"))
    total = len(archivos)

    centros = []
    for i in range(len(archivos)):
        mascara = mascaras.get(i)
        if mascara is not None and mascara.any():
            limpia = limpiar_mascara(mascara)
            ys, xs = np.where(limpia)
            centros.append((xs.mean(), ys.mean()))
        else:
            centros.append(centros[-1] if centros else None)

    centros_suaves = []
    for i in range(len(centros)):
        ini = max(0, i - suavizado)
        fin = min(len(centros), i + suavizado + 1)
        ventana = [c for c in centros[ini:fin] if c is not None]
        if ventana:
            cx = sum(c[0] for c in ventana) / len(ventana)
            cy = sum(c[1] for c in ventana) / len(ventana)
            centros_suaves.append((cx, cy))
        else:
            centros_suaves.append(None)

    for i, archivo in enumerate(archivos):
        frame = cv2.imread(str(archivo))
        h, w = frame.shape[:2]
        centro = centros_suaves[i] or (w / 2, h / 2)

        aspecto_salida = ancho_salida / alto_salida
        crop_h = h
        crop_w = int(h * aspecto_salida)
        if crop_w > w:
            crop_w = w
            crop_h = int(w / aspecto_salida)

        cx, cy = centro
        x1 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
        y1 = int(np.clip(cy - crop_h / 2, 0, h - crop_h))

        recorte = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        recorte = cv2.resize(recorte, (ancho_salida, alto_salida), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(str(Path(carpeta_salida) / f"{i+1:05d}.jpg"), recorte)
        if progreso:
            progreso("autocrop", i + 1, total, f"Recortando... {i+1}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Segmenta y recorta un objeto de un video usando SAM 2")
    parser.add_argument("--video", required=True, help="Ruta al video de entrada")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .pt de SAM 2")
    parser.add_argument("--config", required=True, help="Ruta al archivo de config .yaml del modelo")
    parser.add_argument("--punto", nargs=2, type=int, metavar=("X", "Y"),
                        help="Coordenadas x y del clic sobre el objeto en el primer frame")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Alternativa a --punto: box inicial x1 y1 x2 y2")
    parser.add_argument("--modo", choices=["rotoscopia", "autocrop"], default="autocrop")
    parser.add_argument("--ancho", type=int, default=1080, help="Ancho de salida (solo modo autocrop)")
    parser.add_argument("--alto", type=int, default=1920, help="Alto de salida (solo modo autocrop)")
    parser.add_argument("--salida", default="resultado.mp4", help="Archivo de video final")
    parser.add_argument("--softness", type=int, default=8,
                        help="Ancho de la banda 'desconocido' del trimap en px (0-15)")
    parser.add_argument("--matting", action="store_true",
                        help="Refina bordes con matting por trimap (más lento, borde más limpio)")
    args = parser.parse_args()

    if not args.punto and not args.box:
        sys.exit("Debes dar --punto X Y o --box X1 Y1 X2 Y2 para indicar qué objeto segmentar.")

    trabajo = Path("sam2_trabajo_tmp")
    carpeta_frames = trabajo / "frames"
    carpeta_procesados = trabajo / "procesados"
    audio_path = trabajo / "audio.aac"

    if trabajo.exists():
        shutil.rmtree(trabajo)
    trabajo.mkdir()

    print("1/4 Extrayendo frames...")
    fps = extraer_frames(args.video, str(carpeta_frames))

    print("2/4 Extrayendo audio original...")
    tiene_audio = obtener_audio(args.video, str(audio_path))

    print("3/4 Cargando SAM 2 y segmentando/rastreando el objeto...")
    predictor = cargar_predictor(args.checkpoint, args.config)
    mascaras = segmentar_video(
        predictor, str(carpeta_frames),
        punto_xy=tuple(args.punto) if args.punto else None,
        box_xyxy=tuple(args.box) if args.box else None,
    )

    print(f"4/4 Exportando en modo '{args.modo}'...")
    if args.modo == "rotoscopia":
        exportar_rotoscopia(str(carpeta_frames), mascaras, str(carpeta_procesados),
                            softness=args.softness, matting=args.matting)
        salida_final = frames_a_video(str(carpeta_procesados), args.salida, fps,
                       audio_path=str(audio_path) if tiene_audio else None, con_alpha=True)
    else:
        exportar_autocrop(str(carpeta_frames), mascaras, str(carpeta_procesados),
                          ancho_salida=args.ancho, alto_salida=args.alto,
                          softness=args.softness, matting=args.matting)
        frames_a_video(str(carpeta_procesados), args.salida, fps,
                       audio_path=str(audio_path) if tiene_audio else None, con_alpha=False)

    print(f"\nListo -> {args.salida}")


if __name__ == "__main__":
    main()
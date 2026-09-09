# SAM 2 - Anime Roto Test Runner

Herramienta local de rotoscopia con SAM 2 (Meta) para cortar sujetos de video y exportarlos con fondo transparente.

## Qué incluye

- `anime_roto_runner.py` — App "Anime Roto Test Runner": elegir clip → extraer frames → seleccionar sujeto (vista previa azul) → rastrear → revisar/corregir → exportar.
  - **Rotoscopia sin fondo**: `.mov` ProRes 4444 con canal alpha.
  - **After Effects**: secuencia PNG con alpha.
  - Vista previa con overlay, corrección por pincel y selector de modelo (Rápido Tiny / Equilibrado Base+ / Calidad Large).
- `sam2_gui.py` — GUI de segmentación/recorte (autocrop 9:16 y rotoscopia) con selector de modelo y refinado de bordes (matting opcional).
- `sam2_video_tool.py` — Motor: extracción de frames, segmentación/tracking, limpieza morfológica, matting por trimap, exportación con alpha.

## Requisitos

- Windows, NVIDIA GPU (CUDA) recomendada, ffmpeg en PATH.
- Python 3.12 y:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e sam2-main            # requiere clonar facebookresearch/sam2 en sam2-main/
pip install opencv-python pillow numpy
```

- Clonar SAM 2 y descargar checkpoints:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2-main
# checkpoints en sam2-main/checkpoints/:
#   sam2.1_hiera_tiny.pt, sam2.1_hiera_base_plus.pt, sam2.1_hiera_large.pt
#   (https://dl.fbaipublicfiles.com/segment_anything_2/092824/...)
```

## Uso

```bash
python anime_roto_runner.py
python sam2_gui.py
```

Crea `AnimeRoto.exe` en el escritorio (lanzador) con PyInstaller:

```bash
python -m PyInstaller --noconfirm --onefile --windowed --name AnimeRoto launcher_roto.py
```

Los datos de trabajo se guardan en `anime_roto_data/<clip>/` (frames, máscaras, exportados) — no se suben al repo.
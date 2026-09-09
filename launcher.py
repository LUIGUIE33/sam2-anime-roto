"""
launcher.py
--------------------------------------------------------------------
Pequeño lanzador compilado con PyInstaller que abre la interfaz
SAM 2 usando el Python instalado en el sistema. Se puede copiar a
cualquier lado (ej. el escritorio).
"""

import os
import subprocess
import sys

PROYECTO = r"C:\Users\ESCOBAR\Documents\Default Project\sam2"
GUI = os.path.join(PROYECTO, "sam2_gui.py")


def _aviso(msg):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, "SAM 2 GUI", 0x10)
    except Exception:
        pass


def _intenta_comando(ver):
    try:
        ruta_python = subprocess.run(
            ["where", ver], capture_output=True, text=True, timeout=10).stdout.splitlines()
        if not ruta_python:
            return False
        python = ruta_python[0].strip()
        if python.lower().endswith("pythonw.exe"):
            return python
        # usa pythonw para no abrir consola si está al lado del python
        pyw = python[:-len("python.exe")] + "pythonw.exe"
        return pyw if os.path.exists(pyw) else python
    except Exception:
        return False


def main():
    if not os.path.exists(GUI):
        _aviso(f"No se encontró la interfaz:\n{GUI}\n\nMueve la carpeta sam2 a su ruta original.")
        return

    python = _intenta_comando("pythonw") or _intenta_comando("python")
    if not python:
        _aviso("No se encontró Python en el sistema. Instálalo desde python.org")
        return

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen([python, "-u", GUI], cwd=PROYECTO, creationflags=flags)
    except Exception as e:
        _aviso(f"No se pudo abrir la interfaz:\n{e}")


if __name__ == "__main__":
    sys.exit(main())
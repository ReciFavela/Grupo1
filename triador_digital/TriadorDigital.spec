# -*- mode: python ; coding: utf-8 -*-
# Receita do PyInstaller: transforma o triador.py num programa .exe
# Usado pelo gerar_executavel.bat (não precisa mexer aqui).
from PyInstaller.utils.hooks import collect_dynamic_libs

a = Analysis(
    ["triador.py"],
    pathex=[],
    binaries=collect_dynamic_libs("onnxruntime"),
    datas=[("modelo", "modelo"),  # classificador_pet.onnx (ResNet18) + detector_yolov8n.onnx (YOLO)
           ("imagens_esteira", "imagens_esteira")],
    hiddenimports=["cv2", "PIL.ImageTk"],
    excludes=["torch", "torchvision", "matplotlib", "pandas", "scipy", "IPython", "notebook"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TriadorDigital",
          console=False, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="TriadorDigital")

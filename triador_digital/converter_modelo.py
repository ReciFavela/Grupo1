"""
converter_modelo.py - Converte o modelo treinado (models/best_model.pth, PyTorch)
para o formato ONNX usado pelo Triador Digital (modelo/classificador_pet.onnx).

Por que converter? O arquivo ONNX roda com a biblioteca onnxruntime, que é bem
mais leve que o PyTorch, e o executável fica muito menor.

Quando usar: sempre que o grupo treinar um modelo novo com src/train.py.

Uso (na pasta triador_digital, com torch e torchvision instalados):
    python converter_modelo.py ../models/best_model.pth
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models

origem = Path(sys.argv[1] if len(sys.argv) > 1 else "../models/best_model.pth")
destino = Path(__file__).resolve().parent / "modelo" / "classificador_pet.onnx"
destino.parent.mkdir(exist_ok=True)

# mesma arquitetura do treino: ResNet18 com 2 saídas (NOT_PET, PET)
modelo = models.resnet18(weights=None)
modelo.fc = torch.nn.Linear(modelo.fc.in_features, 2)
modelo.load_state_dict(torch.load(origem, map_location="cpu"))
modelo.eval()

exemplo = torch.randn(1, 3, 224, 224)
torch.onnx.export(modelo, exemplo, str(destino), input_names=["imagem"], output_names=["saida"],
                  dynamic_axes={"imagem": {0: "lote"}, "saida": {0: "lote"}}, opset_version=17,
                  dynamo=False)

# confere se o ONNX responde igual ao PyTorch
import onnxruntime as ort
teste = torch.randn(4, 3, 224, 224)
a = modelo(teste).detach().numpy()
b = ort.InferenceSession(str(destino)).run(None, {"imagem": teste.numpy()})[0]
print(f"Modelo salvo em {destino}")
print(f"Diferença máxima PyTorch x ONNX: {np.abs(a - b).max():.2e} (tem que ser bem pequena)")

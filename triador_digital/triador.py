"""
Triador Digital - ReciFavela (Grupo 1 - PET)
=============================================

Programa que olha a imagem de um resíduo e diz se é garrafa/embalagem PET.

Tem três modos, cada um com um botão grande na tela inicial:

  1. Ver a esteira (simulação): objetos passam numa esteira desenhada na tela.
     Quando passam embaixo da "câmera", o sistema classifica cada um e manda
     para o saco certo (PET, OUTROS ou CONFERIR).
  2. Testar uma foto: a pessoa escolhe uma foto e o sistema diz se é PET.
  3. Usar a câmera: a webcam do computador mostra o objeto e o sistema
     responde ao vivo.

Tudo o que é classificado fica anotado num relatório (arquivo .csv por dia)
com o horário e o turno. A tela "Relatório do dia" mostra os totais.

Como o sistema decide (mesmo fluxo do worker/ingestor.py do projeto):
  1. YOLOv8n (modelo/detector_yolov8n.onnx) acha ONDE estão os objetos na
     imagem (caixas com confiança >= 0,5, igual ao CONFIANCA_MIN do ingestor).
  2. Cada caixa é recortada e vai para a ResNet18 que o grupo treinou
     (modelo/classificador_pet.onnx), que diz se é PET ou NÃO PET.
     Se o YOLO não achar nada, a imagem inteira vai para a ResNet18
     (também igual ao ingestor).
  - Os dois modelos estão em formato ONNX (leve, não precisa do PyTorch).
  - Se a certeza da ResNet18 for menor que LIMITE_CERTEZA, o sistema NÃO
    decide sozinho: pede para uma pessoa conferir (isso é do Triador; o
    ingestor grava sempre a classe mais provável).
  - Na esteira, cada foto já é um objeto só, então ela vai direto para a
    etapa 2 (como se o YOLO já tivesse recortado).

Autores: Grupo 1 - ReciFavela (IMT)
  Arthur Silva Correia, Ilan Hameiry, João Victor Pessoa de Lima dos Anjos,
  Leonardo Souza Olivieri, Vinicius Takashi Nakatsui
"""

import csv
import random
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font as tkfont

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageTk

# ---------------------------------------------------------------------------
# PASTAS
# Quando vira executável (.exe), o PyInstaller guarda o modelo numa pasta
# interna (sys._MEIPASS). As imagens e o relatório ficam ao lado do .exe,
# para qualquer pessoa conseguir trocar as fotos e abrir o relatório.
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    PASTA_APP = Path(sys.executable).parent
    PASTA_RECURSOS = Path(getattr(sys, "_MEIPASS", PASTA_APP))
else:
    PASTA_APP = Path(__file__).resolve().parent
    PASTA_RECURSOS = PASTA_APP

ARQUIVO_MODELO = PASTA_RECURSOS / "modelo" / "classificador_pet.onnx"
ARQUIVO_DETECTOR = PASTA_RECURSOS / "modelo" / "detector_yolov8n.onnx"
PASTA_IMAGENS = PASTA_APP / "imagens_esteira"
if not PASTA_IMAGENS.exists():  # se alguém apagar a pasta, usa a cópia interna
    PASTA_IMAGENS = PASTA_RECURSOS / "imagens_esteira"
PASTA_RELATORIO = PASTA_APP / "relatorio"

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
CLASSES = ["NOT_PET", "PET"]       # mesma ordem usada no treino
LIMITE_CERTEZA = 0.70              # abaixo disso: "pessoa precisa conferir"
CONFIANCA_YOLO = 0.5               # igual ao CONFIANCA_MIN do worker/ingestor.py
IOU_YOLO = 0.7                     # padrão do YOLOv8 para juntar caixas repetidas

# As 80 coisas que o YOLOv8n (treinado no conjunto COCO) sabe encontrar.
NOMES_COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
              'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
# Tradução dos nomes mais comuns numa demonstração
NOMES_PT = {"person": "pessoa", "bottle": "garrafa", "cup": "copo", "wine glass": "taça", "bowl": "pote",
            "cell phone": "celular", "book": "livro", "vase": "vaso", "potted plant": "planta",
            "chair": "cadeira", "laptop": "notebook", "tv": "tela", "keyboard": "teclado",
            "scissors": "tesoura", "handbag": "bolsa", "backpack": "mochila", "remote": "controle",
            "clock": "relógio", "dining table": "mesa", "spoon": "colher", "fork": "garfo", "knife": "faca",
            "toothbrush": "escova", "sports ball": "bola", "suitcase": "mala", "umbrella": "guarda-chuva",
            "teddy bear": "ursinho", "banana": "banana", "apple": "maçã", "orange": "laranja"}
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Normalização ImageNet (igual ao src/train.py e src/predict.py)
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cores (pensadas para projetor: alto contraste)
VERDE = "#1F8A4C"
VERDE_CLARO = "#DDF1E4"
AMARELO = "#E0A100"
AMARELO_CLARO = "#FFF4CC"
CINZA = "#5B6670"
CINZA_CLARO = "#E9ECEE"
VERMELHO = "#C0392B"
FUNDO = "#F7F7F4"
TINTA = "#1E2327"
ESTEIRA = "#2F3439"

# Nomes amigáveis do material, tirados do começo do nome do arquivo de exemplo
# (ex.: "vidro_324.jpg" -> "vidro"). Usado só para explicar os erros.
MATERIAIS = {
    "garrafa_pet": "garrafa/embalagem PET",
    "plastico_outro": "outro tipo de plástico (não PET)",
    "plastico_filme": "plástico filme (não PET)",
    "lata_metal": "lata/metal",
    "vidro": "vidro",
    "papelao": "papelão",
    "papel": "papel",
    "rejeito": "rejeito",
}


# ===========================================================================
# 1) O "CÉREBRO": carrega o modelo e classifica uma imagem
# ===========================================================================
class Classificador:
    """Carrega o modelo ONNX e responde: é PET? com quanta certeza?"""

    def __init__(self, caminho_modelo: Path):
        self.sessao = ort.InferenceSession(str(caminho_modelo), providers=["CPUExecutionProvider"])
        self.entrada = self.sessao.get_inputs()[0].name

    @staticmethod
    def preparar(imagem: Image.Image) -> np.ndarray:
        """Deixa a foto no formato que a rede aprendeu:
        224x224 pixels, cores de 0 a 1, normalizadas, canais na frente."""
        img = imagem.convert("RGB").resize((224, 224), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - MEDIA) / DESVIO
        return x.transpose(2, 0, 1)[None, ...]  # (1, 3, 224, 224)

    def classificar(self, imagem: Image.Image) -> dict:
        """Devolve {'classe': 'PET'/'NOT_PET'/'CONFERIR', 'prob_pet': 0..1, 'certeza': 0..1}."""
        saida = self.sessao.run(None, {self.entrada: self.preparar(imagem)})[0][0]
        e = np.exp(saida - saida.max())
        probs = e / e.sum()                       # softmax: vira porcentagem
        prob_pet = float(probs[1])
        certeza = max(prob_pet, 1 - prob_pet)
        if certeza < LIMITE_CERTEZA:
            classe = "CONFERIR"
        else:
            classe = "PET" if prob_pet >= 0.5 else "NOT_PET"
        return {"classe": classe, "prob_pet": prob_pet, "certeza": certeza}


class Detector:
    """YOLOv8n: encontra os objetos na imagem e devolve uma caixa para cada um.
    É o mesmo yolov8n.pt usado no worker/ingestor.py, convertido para ONNX."""

    TAM = 640  # o YOLOv8n trabalha com imagens de 640x640

    def __init__(self, caminho_modelo: Path):
        self.sessao = ort.InferenceSession(str(caminho_modelo), providers=["CPUExecutionProvider"])
        self.entrada = self.sessao.get_inputs()[0].name

    def detectar(self, imagem: Image.Image, conf=CONFIANCA_YOLO, iou=IOU_YOLO) -> list:
        img = imagem.convert("RGB")
        w, h = img.size
        # "letterbox": diminui a foto sem distorcer e completa com cinza até 640x640
        r = min(self.TAM / w, self.TAM / h)
        nw, nh = round(w * r), round(h * r)
        dx, dy = round((self.TAM - nw) / 2 - 0.1), round((self.TAM - nh) / 2 - 0.1)
        quadro = Image.new("RGB", (self.TAM, self.TAM), (114, 114, 114))
        try:  # OpenCV redimensiona igual ao YOLO original (ultralytics)
            import cv2
            menor = Image.fromarray(cv2.resize(np.asarray(img), (nw, nh), interpolation=cv2.INTER_LINEAR))
        except ImportError:
            menor = img.resize((nw, nh), Image.BILINEAR)
        quadro.paste(menor, (dx, dy))
        x = (np.asarray(quadro, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

        saida = self.sessao.run(None, {self.entrada: x})[0][0].T   # 8400 caixas candidatas x 84
        notas = saida[:, 4:]
        classe = notas.argmax(1)
        nota = notas[np.arange(len(classe)), classe]
        manter = nota >= conf
        saida, classe, nota = saida[manter], classe[manter], nota[manter]
        if len(nota) == 0:
            return []
        # centro/largura/altura (no 640x640) -> cantos na foto original
        cx, cy, bw, bh = saida[:, 0], saida[:, 1], saida[:, 2], saida[:, 3]
        caixas = np.stack([(cx - bw / 2 - dx) / r, (cy - bh / 2 - dy) / r,
                           (cx + bw / 2 - dx) / r, (cy + bh / 2 - dy) / r], axis=1)
        caixas[:, [0, 2]] = caixas[:, [0, 2]].clip(0, w)
        caixas[:, [1, 3]] = caixas[:, [1, 3]].clip(0, h)
        # tira caixas repetidas do mesmo objeto (NMS), separando por tipo de objeto
        indices = self._nms(caixas + classe[:, None] * 4096.0, nota, iou)
        resultado = []
        for i in indices:
            x1, y1, x2, y2 = map(int, caixas[i])
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            nome = NOMES_COCO[int(classe[i])]
            resultado.append({"bbox": [x1, y1, x2, y2], "conf_yolo": round(float(nota[i]), 4),
                              "objeto": NOMES_PT.get(nome, nome)})
        return resultado

    @staticmethod
    def _nms(caixas, notas, iou):
        ordem = notas.argsort()[::-1]
        areas = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
        manter = []
        while ordem.size:
            i = ordem[0]
            manter.append(i)
            xx1 = np.maximum(caixas[i, 0], caixas[ordem[1:], 0])
            yy1 = np.maximum(caixas[i, 1], caixas[ordem[1:], 1])
            xx2 = np.minimum(caixas[i, 2], caixas[ordem[1:], 2])
            yy2 = np.minimum(caixas[i, 3], caixas[ordem[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            sobreposicao = inter / (areas[i] + areas[ordem[1:]] - inter + 1e-9)
            ordem = ordem[1:][sobreposicao <= iou]
        return manter


def analisar_imagem(classificador: Classificador, detector, imagem: Image.Image) -> list:
    """O fluxo do projeto (worker/ingestor.py, função processar_frame):
    YOLO acha os objetos -> cada recorte vai para a ResNet18.
    Se o YOLO não achar nada, a ResNet18 olha a imagem inteira."""
    imagem = imagem.convert("RGB")
    caixas = detector.detectar(imagem) if detector is not None else []
    if not caixas:
        r = classificador.classificar(imagem)
        return [{"bbox": None, "conf_yolo": None, "objeto": "imagem inteira", "recorte": imagem, **r}]
    for c in caixas:
        c["recorte"] = imagem.crop(c["bbox"])
        c.update(classificador.classificar(c["recorte"]))
    return caixas


def deteccao_principal(deteccoes: list) -> dict:
    """Qual resultado mostrar em letras grandes: o maior objeto que não seja
    uma pessoa (quem segura a garrafa costuma aparecer inteiro na câmera)."""
    def area(d):
        if d["bbox"] is None:
            return 0
        x1, y1, x2, y2 = d["bbox"]
        return (x2 - x1) * (y2 - y1)
    sem_pessoa = [d for d in deteccoes if d["objeto"] != "pessoa"]
    return max(sem_pessoa or deteccoes, key=area)


def desenhar_deteccoes(img: Image.Image, deteccoes: list, fonte, grossura=5):
    """Pinta uma caixa colorida em cada objeto, com o resultado escrito em cima."""
    d = ImageDraw.Draw(img)
    for det in deteccoes:
        if det["bbox"] is None:
            continue
        x1, y1, x2, y2 = det["bbox"]
        cor = cor_resultado(det["classe"])
        rotulo = {"PET": "PET", "NOT_PET": "NÃO PET", "CONFERIR": "CONFERIR"}[det["classe"]]
        # só o resultado da ResNet: o nome que o YOLO dá ao objeto não é usado
        # pelo projeto (o ingestor só usa a caixa) e às vezes confunde
        texto = f"{rotulo} {100 * det['certeza']:.0f}%"
        d.rectangle([x1, y1, x2, y2], outline=cor, width=grossura)
        larg = d.textbbox((0, 0), texto, font=fonte)[2]
        alt = d.textbbox((0, 0), texto, font=fonte)[3]
        topo = max(0, y1 - alt - 12)
        d.rectangle([x1, topo, x1 + larg + 16, topo + alt + 12], fill=cor)
        d.text((x1 + 8, topo + 5), texto, fill="white", font=fonte)
    return img


def resumo_deteccoes(deteccoes: list) -> str:
    if len(deteccoes) == 1 and deteccoes[0]["bbox"] is None:
        return "O YOLO não achou objetos: a ResNet olhou a imagem inteira."
    n = len(deteccoes)
    pet = sum(d["classe"] == "PET" for d in deteccoes)
    outros = sum(d["classe"] == "NOT_PET" for d in deteccoes)
    conf = n - pet - outros
    txt = f"{n} objeto{'s' if n > 1 else ''} na imagem: {pet} PET, {outros} não PET"
    return txt + (f", {conf} para conferir" if conf else "")


# ===========================================================================
# 2) RELATÓRIO: anota cada classificação num CSV (um arquivo por dia)
# ===========================================================================
def turno_atual(agora=None) -> str:
    h = (agora or datetime.now()).hour
    if 5 <= h < 12:
        return "manhã"
    if 12 <= h < 18:
        return "tarde"
    return "noite"


class Relatorio:
    CAMPOS = ["data", "hora", "turno", "origem", "arquivo", "resultado", "certeza_%"]

    def __init__(self, pasta: Path):
        self.pasta = pasta
        self.pasta.mkdir(parents=True, exist_ok=True)

    def arquivo_do_dia(self, dia=None) -> Path:
        dia = dia or datetime.now()
        return self.pasta / f"relatorio_{dia:%Y-%m-%d}.csv"

    def anotar(self, origem: str, arquivo: str, r: dict):
        agora = datetime.now()
        caminho = self.arquivo_do_dia(agora)
        novo = not caminho.exists()
        try:
            # utf-8-sig: abre certinho no Excel em português
            with open(caminho, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                if novo:
                    w.writerow(self.CAMPOS)
                w.writerow([f"{agora:%d/%m/%Y}", f"{agora:%H:%M:%S}", turno_atual(agora), origem,
                            arquivo, nome_resultado(r["classe"]), f"{100 * r['certeza']:.0f}"])
        except PermissionError:
            pass  # arquivo aberto no Excel: não trava o programa, só não anota

    def totais_do_dia(self) -> dict:
        """{turno: {'PET': n, 'OUTROS': n, 'CONFERIR': n}}"""
        tot = {t: {"PET": 0, "OUTROS": 0, "CONFERIR": 0} for t in ("manhã", "tarde", "noite")}
        caminho = self.arquivo_do_dia()
        if caminho.exists():
            with open(caminho, encoding="utf-8-sig") as f:
                for linha in csv.DictReader(f, delimiter=";"):
                    t = linha.get("turno")
                    res = linha.get("resultado", "")
                    if t in tot:
                        chave = "PET" if res == "PET" else "CONFERIR" if res.startswith("CONFERIR") else "OUTROS"
                        tot[t][chave] += 1
        return tot


def nome_resultado(classe: str) -> str:
    return {"PET": "PET", "NOT_PET": "NÃO É PET", "CONFERIR": "CONFERIR (sem certeza)"}[classe]


def cor_resultado(classe: str) -> str:
    return {"PET": VERDE, "NOT_PET": CINZA, "CONFERIR": AMARELO}[classe]


def listar_imagens(pasta: Path) -> list:
    """Lista as fotos da pasta. Se a foto estiver em 'pet' ou 'nao_pet',
    a gente sabe a resposta certa e pode mostrar se o sistema acertou."""
    itens = []
    if pasta.exists():
        for f in sorted(pasta.rglob("*")):
            if f.suffix.lower() in EXTENSOES:
                pai = f.parent.name.lower()
                # aceita os nomes do app (pet / nao_pet) e os do dataset do grupo (PET / NOT_PET)
                verdade = "PET" if pai == "pet" else "NOT_PET" if pai in ("nao_pet", "not_pet") else None
                itens.append((f, verdade))
    return itens


def fonte_imagem(tamanho: int):
    """Fonte para escrever por cima do vídeo (Arial no Windows, DejaVu no Linux)."""
    from PIL import ImageFont
    for nome in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:
        return ImageFont.load_default()


def material_do_arquivo(nome: str) -> str:
    for prefixo in sorted(MATERIAIS, key=len, reverse=True):
        if nome.lower().startswith(prefixo):
            return MATERIAIS[prefixo]
    return ""


# ===========================================================================
# 3) A JANELA DO PROGRAMA
# ===========================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Triador Digital - ReciFavela")
        self.configure(bg=FUNDO)
        self.minsize(1100, 680)
        try:
            self.state("zoomed")  # abre maximizado no Windows
        except tk.TclError:
            self.geometry("1280x760")

        familia = "Segoe UI" if "Segoe UI" in tkfont.families() else "DejaVu Sans"
        self.f_titulo = (familia, 30, "bold")
        self.f_grande = (familia, 22, "bold")
        self.f_medio = (familia, 16, "bold")
        self.f_texto = (familia, 14)
        self.f_pequeno = (familia, 11)
        self.familia = familia

        self.relatorio = Relatorio(PASTA_RELATORIO)
        try:
            self.modelo = Classificador(ARQUIVO_MODELO)
        except Exception as erro:  # sem modelo não há o que fazer
            self.modelo = None
            self.erro_modelo = str(erro)
        try:
            self.detector = Detector(ARQUIVO_DETECTOR)
        except Exception:  # sem o YOLO, funciona só com a etapa 2 (ResNet)
            self.detector = None

        self.tela = None
        self.mostrar(TelaInicio)

    def mostrar(self, classe_tela, **kw):
        """Troca a tela atual por outra."""
        if self.tela is not None:
            self.tela.fechar()
            self.tela.destroy()
        self.tela = classe_tela(self, **kw)
        self.tela.pack(fill="both", expand=True)

    def botao(self, pai, texto, comando, cor=VERDE, fonte=None, largura=None, **kw):
        """Botão grande e fácil de clicar."""
        b = tk.Button(pai, text=texto, command=comando, font=fonte or self.f_medio,
                      bg=cor, fg="white", activebackground=cor, activeforeground="white",
                      relief="flat", bd=0, cursor="hand2", padx=22, pady=12, **kw)
        if largura:
            b.configure(width=largura)
        return b


class Tela(tk.Frame):
    def __init__(self, app: App):
        super().__init__(app, bg=FUNDO)
        self.app = app

    def fechar(self):
        """Chamado antes de sair da tela (para parar animação/câmera)."""

    def barra_topo(self, titulo: str):
        barra = tk.Frame(self, bg="white", height=70)
        barra.pack(fill="x")
        self.app.botao(barra, "⟵  Voltar", lambda: self.app.mostrar(TelaInicio),
                       cor=CINZA, fonte=self.app.f_texto).pack(side="left", padx=14, pady=10)
        tk.Label(barra, text=titulo, font=self.app.f_grande, bg="white", fg=TINTA).pack(side="left", padx=18)
        return barra


# ---------------------------------------------------------------------------
# TELA INICIAL
# ---------------------------------------------------------------------------
class TelaInicio(Tela):
    def __init__(self, app):
        super().__init__(app)
        tk.Label(self, text="Triador Digital", font=app.f_titulo, bg=FUNDO, fg=VERDE).pack(pady=(38, 0))
        tk.Label(self, text="O computador olha o material e diz se é garrafa PET",
                 font=app.f_texto, bg=FUNDO, fg=CINZA).pack(pady=(4, 6))
        tk.Label(self, text=f"Turno agora: {turno_atual()}", font=app.f_pequeno, bg=FUNDO, fg=CINZA).pack()

        if app.modelo is None:
            tk.Label(self, text="Não encontrei o arquivo do modelo.\nVeja o README, parte 'Deu problema?'.\n\n"
                     + app.erro_modelo, font=app.f_texto, bg=FUNDO, fg=VERMELHO).pack(pady=40)
            return

        caixa = tk.Frame(self, bg=FUNDO)
        caixa.pack(pady=28)
        opcoes = [
            ("1", "Ver a esteira", "Simulação: os objetos passam\ne o sistema separa sozinho",
             lambda: app.mostrar(TelaEsteira)),
            ("2", "Testar uma foto", "Escolha uma foto do computador\ne veja a resposta",
             lambda: app.mostrar(TelaFoto)),
            ("3", "Usar a câmera", "Mostre um objeto para a câmera\ndo computador",
             lambda: app.mostrar(TelaCamera)),
        ]
        for i, (num, titulo, desc, cmd) in enumerate(opcoes):
            card = tk.Frame(caixa, bg="white", highlightbackground="#D9DEE1", highlightthickness=1,
                            cursor="hand2")
            card.grid(row=0, column=i, padx=16, ipadx=10, ipady=10)
            tk.Label(card, text=num, font=(app.familia, 40, "bold"), bg="white", fg=VERDE).pack(pady=(16, 0))
            tk.Label(card, text=titulo, font=app.f_grande, bg="white", fg=TINTA).pack()
            tk.Label(card, text=desc, font=app.f_texto, bg="white", fg=CINZA, justify="center").pack(pady=10)
            app.botao(card, "Abrir", cmd, largura=12).pack(pady=(4, 18))
            for w in (card,) + tuple(card.winfo_children()[:3]):
                w.bind("<Button-1>", lambda e, c=cmd: c())

        rodape = tk.Frame(self, bg=FUNDO)
        rodape.pack(pady=10)
        app.botao(rodape, "Relatório do dia", lambda: app.mostrar(TelaRelatorio),
                  cor=TINTA, fonte=app.f_texto).pack(side="left", padx=8)
        app.botao(rodape, "Sair", app.destroy, cor=CINZA, fonte=app.f_texto).pack(side="left", padx=8)
        tk.Label(self, text="ReciFavela · Grupo 1 (PET) · Instituto Mauá de Tecnologia",
                 font=app.f_pequeno, bg=FUNDO, fg=CINZA).pack(side="bottom", pady=12)


# ---------------------------------------------------------------------------
# PAINEL DE RESULTADO (usado nas três telas)
# ---------------------------------------------------------------------------
class PainelResultado(tk.Frame):
    """Mostra a foto que o sistema analisou e a resposta em letras grandes."""

    def __init__(self, pai, app: App, largura=400):
        super().__init__(pai, bg="white", width=largura)
        self.app = app
        self.largura = largura
        self.pack_propagate(False)
        tk.Label(self, text="O que o sistema viu", font=app.f_medio, bg="white", fg=CINZA).pack(pady=(16, 6))
        self.alt_foto = min(largura - 120, 280)  # cabe em telas de 768 px de altura
        moldura = tk.Frame(self, bg=CINZA_CLARO, width=largura - 60, height=self.alt_foto)
        moldura.pack(padx=20)
        moldura.pack_propagate(False)  # mantém o tamanho mesmo sem foto
        self.foto = tk.Label(moldura, bg=CINZA_CLARO)
        self.foto.pack(fill="both", expand=True)
        self.resposta = tk.Label(self, text="—", font=(app.familia, 34, "bold"), bg="white", fg=CINZA)
        self.resposta.pack(pady=(14, 0))
        self.barra = tk.Canvas(self, width=largura - 60, height=26, bg="white", highlightthickness=0)
        self.barra.pack(pady=6)
        self.certeza = tk.Label(self, text="", font=app.f_texto, bg="white", fg=TINTA)
        self.certeza.pack()
        self.explica = tk.Label(self, text="", font=app.f_texto, bg="white", fg=TINTA,
                                wraplength=largura - 50, justify="center")
        self.explica.pack(pady=(10, 0))
        self.acerto = tk.Label(self, text="", font=app.f_medio, bg="white", fg=VERDE,
                               wraplength=largura - 40, justify="center")
        self.acerto.pack(pady=(10, 0))
        self.resumo = tk.Label(self, text="", font=app.f_pequeno, bg="white", fg=CINZA,
                               wraplength=largura - 40, justify="center")
        self.resumo.pack(pady=(8, 0))
        self._tk_img = None

    def mostrar(self, imagem: Image.Image, r: dict, verdade=None, material=""):
        lado = self.largura - 60
        img = imagem.convert("RGB").copy()
        img.thumbnail((lado, self.alt_foto))
        self._tk_img = ImageTk.PhotoImage(img)
        self.foto.configure(image=self._tk_img)

        cor = cor_resultado(r["classe"])
        self.resposta.configure(text={"PET": "É PET", "NOT_PET": "NÃO É PET", "CONFERIR": "CONFERIR"}[r["classe"]],
                                fg=cor)
        # barra de certeza
        w = self.largura - 60
        self.barra.delete("all")
        self.barra.create_rectangle(0, 0, w, 26, fill=CINZA_CLARO, outline="")
        self.barra.create_rectangle(0, 0, w * r["certeza"], 26, fill=cor, outline="")
        self.certeza.configure(text=f"Certeza: {100 * r['certeza']:.0f}%")

        if r["classe"] == "PET":
            txt = "Vai para o saco do PET."
        elif r["classe"] == "NOT_PET":
            txt = "Não vai para o PET.\nSegue para a separação dos outros materiais."
        else:
            txt = "O sistema não tem certeza.\nUma pessoa precisa olhar e decidir."
        self.explica.configure(text=txt)

        if verdade is None or r["classe"] == "CONFERIR":
            self.acerto.configure(text="")
        elif r["classe"] == verdade:
            self.acerto.configure(text="✓ Acertou", fg=VERDE)
        else:
            era = f"Isto é {material}." if material else ("Isto é PET." if verdade == "PET" else "Isto não é PET.")
            self.acerto.configure(text=f"✗ O sistema errou. {era}", fg=VERMELHO)

    def limpar(self, texto="—"):
        self.foto.configure(image="")
        self.resposta.configure(text=texto, fg=CINZA)
        self.barra.delete("all")
        for w in (self.certeza, self.explica, self.acerto, self.resumo):
            w.configure(text="")


# ---------------------------------------------------------------------------
# TELA 1: SIMULAÇÃO DA ESTEIRA
# ---------------------------------------------------------------------------
class Item:
    """Um objeto andando na esteira."""

    def __init__(self, caminho: Path, verdade, imagem: Image.Image, tk_img, x, y):
        self.caminho = caminho
        self.verdade = verdade
        self.imagem = imagem
        self.tk_img = tk_img
        self.x, self.y = x, y
        self.resultado = None        # preenchido quando passa na câmera
        self.caindo = False          # True quando sai da esteira e vai pro saco
        self.alvo = None             # (x, y) do saco de destino
        self.id_img = None
        self.id_txt = None
        self.id_etiqueta = None
        self.id_borda = None


class TelaEsteira(Tela):
    TAM = 120          # tamanho de cada objeto na tela (pixels)
    ESPACO = 210       # distância entre um objeto e outro
    VELOCIDADES = [("Devagar", 2), ("Normal", 4), ("Rápido", 7)]

    def __init__(self, app):
        super().__init__(app)
        barra = self.barra_topo("Esteira (simulação)")
        self.btn_play = app.botao(barra, "▶  Começar", self.alternar, fonte=app.f_texto)
        self.btn_play.pack(side="right", padx=14)
        self.vel_idx = 0
        self.btn_vel = app.botao(barra, "Velocidade: Devagar", self.trocar_velocidade, cor=TINTA,
                                 fonte=app.f_texto)
        self.btn_vel.pack(side="right", padx=4)
        self.placar = tk.Label(barra, text="", font=app.f_texto, bg="white", fg=TINTA)
        self.placar.pack(side="right", padx=20)

        corpo = tk.Frame(self, bg=FUNDO)
        corpo.pack(fill="both", expand=True)
        self.painel = PainelResultado(corpo, app)
        self.painel.pack(side="right", fill="y", padx=(0, 14), pady=14)
        self.canvas = tk.Canvas(corpo, bg=FUNDO, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=14, pady=14)

        self.fila = listar_imagens(PASTA_IMAGENS)
        random.shuffle(self.fila)
        self.pos_fila = 0
        self.itens = []
        self.rodando = False
        self.deslocamento = 0      # para animar as listras da esteira
        self.contagem = {"PET": 0, "NOT_PET": 0, "CONFERIR": 0}
        self.acertos = 0
        self.avaliados = 0
        self.analisados = 0
        self.flash = 0
        self.flash_cor = VERDE
        self._after = None

        if not self.fila:
            self.painel.limpar("Sem fotos")
            self.painel.explica.configure(text=f"Coloque fotos na pasta:\n{PASTA_IMAGENS}")
        else:
            self.painel.limpar("Aperte Começar")
        self.atualizar_placar()
        self.canvas.bind("<Configure>", lambda e: self.desenhar())
        self.after(50, self.desenhar)

    # --- geometria (calculada a partir do tamanho da janela) ---
    def geo(self):
        W = max(self.canvas.winfo_width(), 600)
        H = max(self.canvas.winfo_height(), 450)
        y_esteira = int(H * 0.30)
        alt_esteira = self.TAM + 40
        fim_esteira = int(W * 0.80)
        x_camera = int(W * 0.42)
        sacos = {}
        larg_saco = int(W * 0.26)
        for i, k in enumerate(["PET", "CONFERIR", "NOT_PET"]):
            cx = int(W * (0.18 + 0.32 * i))
            sacos[k] = (cx - larg_saco // 2, H - 170, cx + larg_saco // 2, H - 10)
        return W, H, y_esteira, alt_esteira, fim_esteira, x_camera, sacos

    def desenhar(self):
        """Desenha o cenário fixo: esteira, câmera e sacos."""
        c = self.canvas
        c.delete("cenario")
        W, H, ye, ae, fim, xc, sacos = self.geo()
        f = self.app.familia
        # esteira com listras andando
        c.create_rectangle(0, ye, fim, ye + ae, fill=ESTEIRA, outline="", tags="cenario")
        for x in range(-60 + self.deslocamento % 60, fim, 60):
            c.create_line(x, ye + 4, x + 20, ye + ae - 4, fill="#3E454B", width=6, tags="cenario")
        c.create_rectangle(0, ye + ae, fim, ye + ae + 10, fill="#1A1D20", outline="", tags="cenario")
        c.create_text(10, ye + ae + 28, text="ESTEIRA  →", anchor="w", font=(f, 12, "bold"), fill=CINZA,
                      tags="cenario")
        # câmera em cima da esteira, com o "campo de visão"
        cor_cam = self.flash_cor if self.flash > 0 else TINTA
        c.create_polygon(xc - 22, ye - 50, xc + 22, ye - 50, xc + 80, ye - 4, xc - 80, ye - 4,
                         fill="#FFFFFF", outline=cor_cam, width=3 if self.flash else 2, dash=(6, 4),
                         tags="cenario")
        c.create_rectangle(xc - 44, ye - 104, xc + 44, ye - 52, fill=TINTA, outline="", tags="cenario")
        c.create_oval(xc - 16, ye - 94, xc + 16, ye - 62, fill="#3D7BB8", outline="white", width=3,
                      tags="cenario")
        c.create_text(xc, ye - 120, text="CÂMERA", font=(f, 13, "bold"), fill=TINTA, tags="cenario")
        # sacos
        info = {"PET": ("SACO DO PET", VERDE, VERDE_CLARO),
                "CONFERIR": ("PESSOA CONFERE", AMARELO, AMARELO_CLARO),
                "NOT_PET": ("OUTROS MATERIAIS", CINZA, CINZA_CLARO)}
        for k, (x1, y1, x2, y2) in sacos.items():
            nome, cor, claro = info[k]
            c.create_rectangle(x1, y1, x2, y2, fill=claro, outline=cor, width=3, tags="cenario")
            c.create_text((x1 + x2) // 2, y1 + 28, text=nome, font=(f, 15, "bold"), fill=cor, tags="cenario")
            c.create_text((x1 + x2) // 2, y1 + 90, text=str(self.contagem[k]), font=(f, 46, "bold"),
                          fill=cor, tags="cenario")
        c.tag_lower("cenario")

    # --- controles ---
    def alternar(self):
        if not self.fila:
            return
        self.rodando = not self.rodando
        self.btn_play.configure(text="⏸  Pausar" if self.rodando else "▶  Continuar",
                                bg=AMARELO if self.rodando else VERDE)
        if self.rodando:
            self.passo()

    def trocar_velocidade(self):
        self.vel_idx = (self.vel_idx + 1) % len(self.VELOCIDADES)
        self.btn_vel.configure(text=f"Velocidade: {self.VELOCIDADES[self.vel_idx][0]}")

    def atualizar_placar(self):
        txt = f"Analisados: {self.analisados}"
        if self.avaliados:
            txt += f"    Acertou: {self.acertos} de {self.avaliados}"
        self.placar.configure(text=txt)

    # --- animação ---
    def novo_item(self, x):
        caminho, verdade = self.fila[self.pos_fila % len(self.fila)]
        self.pos_fila += 1
        if self.pos_fila % len(self.fila) == 0:
            random.shuffle(self.fila)
        try:
            img = Image.open(caminho).convert("RGB")
        except Exception:
            return
        mini = img.copy()
        mini.thumbnail((self.TAM, self.TAM))
        quadro = Image.new("RGB", (self.TAM, self.TAM), "white")
        quadro.paste(mini, ((self.TAM - mini.width) // 2, (self.TAM - mini.height) // 2))
        _, _, ye, _, _, _, _ = self.geo()
        it = Item(caminho, verdade, img, ImageTk.PhotoImage(quadro), x, ye + 20)
        it.id_img = self.canvas.create_image(it.x, it.y, image=it.tk_img, anchor="nw")
        it.id_borda = self.canvas.create_rectangle(it.x, it.y, it.x + self.TAM, it.y + self.TAM,
                                                   outline="white", width=2)
        self.itens.append(it)

    def passo(self):
        """Um 'quadro' da animação (roda ~33 vezes por segundo)."""
        if not self.rodando:
            return
        v = self.VELOCIDADES[self.vel_idx][1]
        W, H, ye, ae, fim, xc, sacos = self.geo()
        self.deslocamento += v

        # coloca objeto novo quando há espaço no começo da esteira
        na_esteira = [i for i in self.itens if not i.caindo]
        if not na_esteira or min(i.x for i in na_esteira) > self.ESPACO - self.TAM:
            self.novo_item(-self.TAM)

        for it in list(self.itens):
            if not it.caindo:
                it.x += v
                centro = it.x + self.TAM / 2
                # passou embaixo da câmera: classifica!
                if it.resultado is None and centro >= xc:
                    self.classificar_item(it)
                # chegou no fim da esteira: vai para o saco
                if it.x + self.TAM / 2 >= fim:
                    it.caindo = True
                    x1, y1, x2, y2 = sacos[it.resultado["classe"]]
                    it.alvo = ((x1 + x2) / 2 - self.TAM / 2, y1 - self.TAM * 0.55)
            else:
                ax, ay = it.alvo
                dx, dy = ax - it.x, ay - it.y
                d = (dx * dx + dy * dy) ** 0.5
                passo = max(8, v * 3)
                if d <= passo:
                    self.contagem[it.resultado["classe"]] += 1
                    for i in (it.id_img, it.id_txt, it.id_etiqueta, it.id_borda):
                        if i:
                            self.canvas.delete(i)
                    self.itens.remove(it)
                    self.desenhar()
                    continue
                it.x += dx / d * passo
                it.y += dy / d * passo
            self.canvas.coords(it.id_img, it.x, it.y)
            self.canvas.coords(it.id_borda, it.x, it.y, it.x + self.TAM, it.y + self.TAM)
            if it.id_txt:
                self.canvas.coords(it.id_etiqueta, it.x, it.y + self.TAM - 28, it.x + self.TAM, it.y + self.TAM)
                self.canvas.coords(it.id_txt, it.x + self.TAM / 2, it.y + self.TAM - 14)

        if self.flash > 0:
            self.flash -= 1
        if self.deslocamento % 12 < v or self.flash in (0, 1):
            self.desenhar()
        self._after = self.after(30, self.passo)

    def classificar_item(self, it: Item):
        r = self.app.modelo.classificar(it.imagem)
        it.resultado = r
        cor = cor_resultado(r["classe"])
        rotulo = {"PET": "PET", "NOT_PET": "NÃO PET", "CONFERIR": "CONFERIR"}[r["classe"]]
        it.id_etiqueta = self.canvas.create_rectangle(it.x, it.y + self.TAM - 28, it.x + self.TAM,
                                                      it.y + self.TAM, fill=cor, outline=cor)
        it.id_txt = self.canvas.create_text(it.x + self.TAM / 2, it.y + self.TAM - 14,
                                            text=f"{rotulo} {100 * r['certeza']:.0f}%",
                                            font=(self.app.familia, 12, "bold"), fill="white")
        self.canvas.itemconfigure(it.id_borda, outline=cor, width=5)
        self.flash, self.flash_cor = 8, cor
        self.analisados += 1
        material = material_do_arquivo(it.caminho.name)
        self.painel.mostrar(it.imagem, r, it.verdade, material)
        if it.verdade is not None and r["classe"] != "CONFERIR":
            self.avaliados += 1  # só conta acerto quando o sistema decidiu sozinho
            self.acertos += int(r["classe"] == it.verdade)
        self.atualizar_placar()
        self.app.relatorio.anotar("simulação", it.caminho.name, r)

    def fechar(self):
        self.rodando = False
        if self._after:
            self.after_cancel(self._after)


# ---------------------------------------------------------------------------
# TELA 2: TESTAR UMA FOTO
# ---------------------------------------------------------------------------
class TelaFoto(Tela):
    def __init__(self, app):
        super().__init__(app)
        barra = self.barra_topo("Testar uma foto")
        app.botao(barra, "Escolher foto…", self.escolher, fonte=app.f_texto).pack(side="right", padx=14)

        corpo = tk.Frame(self, bg=FUNDO)
        corpo.pack(fill="both", expand=True)
        self.painel = PainelResultado(corpo, app, largura=460)
        self.painel.pack(side="right", fill="y", padx=14, pady=14)
        esquerda = tk.Frame(corpo, bg=FUNDO)
        esquerda.pack(side="left", fill="both", expand=True, padx=14, pady=14)
        self.grande = tk.Label(esquerda, bg=CINZA_CLARO,
                               text="Clique em  “Escolher foto…”  lá em cima\n\n"
                                    "O sistema procura os objetos na foto (YOLO)\n"
                                    "e diz se cada um é PET (ResNet do grupo).",
                               font=app.f_texto, fg=CINZA)
        self.grande.pack(fill="both", expand=True)
        self._tk = None
        self.fonte = fonte_imagem(22)
        self.painel.limpar()

    def escolher(self):
        caminho = filedialog.askopenfilename(
            title="Escolha a foto do objeto",
            initialdir=str(PASTA_IMAGENS if PASTA_IMAGENS.exists() else Path.home()),
            filetypes=[("Fotos", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")])
        if not caminho:
            return
        try:
            img = Image.open(caminho).convert("RGB")
        except Exception:
            self.painel.limpar("Não abriu")
            self.painel.explica.configure(text="Esse arquivo não é uma foto que eu consiga abrir.")
            return
        # fluxo do projeto: YOLO acha os objetos -> ResNet decide cada um
        deteccoes = analisar_imagem(self.app.modelo, self.app.detector, img)
        principal = deteccao_principal(deteccoes)
        p = Path(caminho)
        pai = p.parent.name.lower()
        verdade = "PET" if pai == "pet" else "NOT_PET" if pai in ("nao_pet", "not_pet") else None
        if len(deteccoes) > 1:
            verdade = None  # a pasta diz o que é a foto, não cada objeto dentro dela
        self.painel.mostrar(principal["recorte"], principal, verdade, material_do_arquivo(p.name))
        self.painel.resumo.configure(text=resumo_deteccoes(deteccoes))
        # a fonte acompanha o tamanho da foto, para a etiqueta ficar legível depois de diminuir
        mostra = desenhar_deteccoes(img.copy(), deteccoes, fonte_imagem(max(14, img.width // 40)),
                                    grossura=max(3, img.width // 250))
        mostra.thumbnail((max(self.grande.winfo_width() - 20, 300), max(self.grande.winfo_height() - 20, 300)))
        self._tk = ImageTk.PhotoImage(mostra)
        self.grande.configure(image=self._tk, text="")
        for d in deteccoes:
            self.app.relatorio.anotar("foto", f"{p.name} ({d['objeto']})", d)


# ---------------------------------------------------------------------------
# TELA 3: CÂMERA AO VIVO
# ---------------------------------------------------------------------------
class TelaCamera(Tela):
    INTERVALO_ANALISE = 0.4   # segundos entre uma análise e outra

    def __init__(self, app):
        super().__init__(app)
        barra = self.barra_topo("Câmera ao vivo")
        self.btn_anotar = app.botao(barra, "Anotar no relatório", self.anotar, fonte=app.f_texto)
        self.btn_anotar.pack(side="right", padx=14)
        # dois jeitos de olhar: o fluxo do projeto (padrão) ou só o quadrado do meio (plano B)
        self.modo_projeto = app.detector is not None
        self.btn_modo = app.botao(barra, "", self.trocar_modo, cor=TINTA, fonte=app.f_texto)
        self.btn_modo.pack(side="right", padx=4)
        self.atualizar_botao_modo()
        # webcam USB ou celular (DroidCam, Iriun, Camo…) aparecem como câmera 2, 3…
        self.btn_trocar = app.botao(barra, "Trocar câmera", self.trocar_camera, cor=CINZA,
                                    fonte=app.f_texto)
        self.btn_trocar.pack(side="right", padx=4)

        corpo = tk.Frame(self, bg=FUNDO)
        corpo.pack(fill="both", expand=True)
        self.painel = PainelResultado(corpo, app)
        self.painel.pack(side="right", fill="y", padx=14, pady=14)
        self.video = tk.Label(corpo, bg="black", fg="white", font=app.f_texto,
                              text="Ligando a câmera…")
        self.video.pack(side="left", fill="both", expand=True, padx=14, pady=14)
        self.painel.limpar()

        self.cap = None
        self.indice_camera = -1   # qual câmera está aberta (0 = a primeira do computador)
        self.fonte = fonte_imagem(24)
        self.ultimo = 0.0
        self.historico = []      # últimas probabilidades (deixa a resposta mais estável)
        self.ultimo_resultado = None
        self.ultimas_deteccoes = []
        self._tk = None
        self._after = None
        self.after(100, self.abrir_camera)

    def atualizar_botao_modo(self):
        self.btn_modo.configure(text="Olhando: todos os objetos" if self.modo_projeto
                                else "Olhando: só o quadrado")

    def trocar_modo(self):
        if self.app.detector is None:
            return
        self.modo_projeto = not self.modo_projeto
        self.historico, self.ultimas_deteccoes, self.ultimo_resultado = [], [], None
        self.atualizar_botao_modo()

    MAX_CAMERAS = 6   # procura da câmera 0 até a 5

    def abrir_camera(self, a_partir_de=0):
        """Abre a primeira câmera que funcionar, começando pela de número a_partir_de.
        Se chegar no fim da lista, volta para a câmera 0."""
        try:
            import cv2  # só carrega se a pessoa usar a câmera
        except ImportError:
            self.video.configure(text="O programa foi instalado sem suporte à câmera.")
            return
        self.cv2 = cv2
        for passo in range(self.MAX_CAMERAS):
            indice = (a_partir_de + passo) % self.MAX_CAMERAS
            cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(indice)
            if cap is not None and cap.isOpened() and cap.read()[0]:
                self.cap = cap
                self.indice_camera = indice
                break
            if cap is not None:
                cap.release()
        if self.cap is None:
            self.video.configure(image="", text="Não encontrei nenhuma câmera.\n\n"
                                      "Veja se a webcam está ligada ou conectada no USB\n"
                                      "e se outro programa (Zoom, Teams…) não está usando ela.")
            return
        self.btn_trocar.configure(text=f"Trocar câmera (usando a {self.indice_camera + 1})")
        self.loop()

    def trocar_camera(self):
        """Fecha a câmera atual e abre a próxima que existir."""
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        anterior = self.indice_camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.historico, self.ultimas_deteccoes, self.ultimo_resultado = [], [], None
        self.video.configure(image="", text="Procurando outra câmera…")
        self.update_idletasks()
        self.abrir_camera(anterior + 1)

    def loop(self):
        ok, quadro = self.cap.read()
        if ok:
            quadro = self.cv2.cvtColor(quadro, self.cv2.COLOR_BGR2RGB)
            img = Image.fromarray(quadro)
            if self.modo_projeto:
                self.loop_projeto(img)
            else:
                self.loop_quadrado(img)
        self._after = self.after(30, self.loop)

    def mostrar_video(self, img):
        area_w = max(self.video.winfo_width(), 320)
        area_h = max(self.video.winfo_height(), 240)
        img.thumbnail((area_w, area_h))
        self._tk = ImageTk.PhotoImage(img)
        self.video.configure(image=self._tk, text="")

    def loop_projeto(self, img):
        """Fluxo do projeto: YOLO acha os objetos, a ResNet decide cada um."""
        agora = time.time()
        if agora - self.ultimo >= self.INTERVALO_ANALISE:
            self.ultimo = agora
            self.ultimas_deteccoes = analisar_imagem(self.app.modelo, self.app.detector, img)
            principal = deteccao_principal(self.ultimas_deteccoes)
            self.ultimo_resultado = principal
            self.painel.mostrar(principal["recorte"], principal)
            self.painel.resumo.configure(text=resumo_deteccoes(self.ultimas_deteccoes))
        self.mostrar_video(desenhar_deteccoes(img, self.ultimas_deteccoes, self.fonte))

    def loop_quadrado(self, img):
        """Plano B: o sistema olha só o quadrado do meio (onde a pessoa coloca o objeto)."""
        w, h = img.size
        lado = int(min(w, h) * 0.7)
        x1, y1 = (w - lado) // 2, (h - lado) // 2
        recorte = img.crop((x1, y1, x1 + lado, y1 + lado))

        agora = time.time()
        if agora - self.ultimo >= self.INTERVALO_ANALISE:
            self.ultimo = agora
            r = self.app.modelo.classificar(recorte)
            self.historico = (self.historico + [r["prob_pet"]])[-3:]
            p = sum(self.historico) / len(self.historico)
            certeza = max(p, 1 - p)
            classe = "CONFERIR" if certeza < LIMITE_CERTEZA else ("PET" if p >= 0.5 else "NOT_PET")
            self.ultimo_resultado = {"classe": classe, "prob_pet": p, "certeza": certeza}
            self.ultimas_deteccoes = []
            self.painel.mostrar(recorte, self.ultimo_resultado)
            self.painel.resumo.configure(text="Olhando só o quadrado do meio (sem o YOLO).")

        # desenha o quadrado-guia por cima do vídeo
        cor = cor_resultado(self.ultimo_resultado["classe"]) if self.ultimo_resultado else "#FFFFFF"
        d = ImageDraw.Draw(img)
        d.rectangle([x1, y1, x1 + lado, y1 + lado], outline=cor, width=6)
        aviso = "Coloque o objeto aqui dentro"
        larg_txt = d.textbbox((0, 0), aviso, font=self.fonte)[2]
        d.rectangle([x1, y1 - 40, x1 + larg_txt + 20, y1], fill=cor)
        d.text((x1 + 10, y1 - 36), aviso, fill="white", font=self.fonte)
        self.mostrar_video(img)

    def anotar(self):
        if self.ultimas_deteccoes:
            for d in self.ultimas_deteccoes:
                self.app.relatorio.anotar("câmera", d["objeto"], d)
        elif self.ultimo_resultado:
            self.app.relatorio.anotar("câmera", "quadrado do meio", self.ultimo_resultado)
        if self.ultimo_resultado:
            self.btn_anotar.configure(text="✓ Anotado")
            self.after(1200, lambda: self.btn_anotar.configure(text="Anotar no relatório"))

    def fechar(self):
        if self._after:
            self.after_cancel(self._after)
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# TELA: RELATÓRIO DO DIA
# ---------------------------------------------------------------------------
class TelaRelatorio(Tela):
    def __init__(self, app):
        super().__init__(app)
        barra = self.barra_topo("Relatório de hoje")
        app.botao(barra, "Abrir a pasta do relatório", self.abrir_pasta, cor=TINTA,
                  fonte=app.f_texto).pack(side="right", padx=14)
        tot = app.relatorio.totais_do_dia()
        tk.Label(self, text=f"{datetime.now():%d/%m/%Y}  ·  quantos itens o sistema analisou em cada turno",
                 font=app.f_texto, bg=FUNDO, fg=CINZA).pack(pady=(22, 8))

        grade = tk.Frame(self, bg=FUNDO)
        grade.pack(pady=10)
        maximo = max([1] + [v for t in tot.values() for v in t.values()])
        for col, turno in enumerate(("manhã", "tarde", "noite")):
            card = tk.Frame(grade, bg="white", highlightbackground="#D9DEE1", highlightthickness=1)
            card.grid(row=0, column=col, padx=14, ipadx=14, ipady=8)
            tk.Label(card, text=turno.upper(), font=app.f_grande, bg="white", fg=TINTA).pack(pady=(12, 4))
            cv = tk.Canvas(card, width=300, height=230, bg="white", highlightthickness=0)
            cv.pack(padx=10)
            for i, (k, nome, cor) in enumerate([("PET", "PET", VERDE), ("OUTROS", "Outros", CINZA),
                                                ("CONFERIR", "Conferir", AMARELO)]):
                v = tot[turno][k]
                x = 20 + i * 95
                h = 150 * v / maximo
                cv.create_rectangle(x, 180 - h, x + 70, 180, fill=cor, outline="")
                cv.create_text(x + 35, 170 - h, text=str(v), font=(app.familia, 18, "bold"), fill=cor)
                cv.create_text(x + 35, 202, text=nome, font=app.f_pequeno, fill=TINTA)
        tk.Label(self, text=f"O relatório completo (cada item, com hora e certeza) fica em:\n"
                            f"{app.relatorio.arquivo_do_dia()}\nDá para abrir no Excel.",
                 font=app.f_pequeno, bg=FUNDO, fg=CINZA, justify="center").pack(pady=20)

    def abrir_pasta(self):
        import os
        import subprocess
        p = str(self.app.relatorio.pasta)
        if sys.platform == "win32":
            os.startfile(p)
        else:
            subprocess.Popen(["xdg-open", p])


# ===========================================================================
# MODO TESTE (sem janela): python triador.py --teste
# Classifica todas as fotos de imagens_esteira e mostra o resultado no terminal.
# Serve para conferir se o modelo está funcionando.
# ===========================================================================
def modo_teste():
    modelo = Classificador(ARQUIVO_MODELO)
    itens = listar_imagens(PASTA_IMAGENS)
    acertos = avaliados = conferir = 0
    print(f"Modelo: {ARQUIVO_MODELO}\nFotos:  {PASTA_IMAGENS} ({len(itens)} fotos)\n")
    for caminho, verdade in itens:
        r = modelo.classificar(Image.open(caminho))
        marca = ""
        if r["classe"] == "CONFERIR":
            conferir += 1
        elif verdade:
            avaliados += 1
            ok = r["classe"] == verdade
            acertos += ok
            marca = "ok" if ok else "ERROU"
        print(f"{caminho.parent.name:8s} {caminho.name:28s} -> {nome_resultado(r['classe']):22s} "
              f"{100 * r['certeza']:5.1f}%  {marca}")
    print(f"\nAcertos: {acertos} de {avaliados}  ·  mandou conferir: {conferir}")
    if ARQUIVO_DETECTOR.exists():
        detector = Detector(ARQUIVO_DETECTOR)
        com_objeto = 0
        for caminho, _ in itens:
            com_objeto += bool(detector.detectar(Image.open(caminho)))
        print(f"YOLO (etapa 1): achou objeto em {com_objeto} de {len(itens)} fotos")
    else:
        print("YOLO (etapa 1): arquivo do detector não encontrado")


if __name__ == "__main__":
    if "--teste" in sys.argv:
        modo_teste()
    else:
        App().mainloop()

"""
baixar_fotos_taco.py - Baixa fotos do TACO (Trash Annotations in Context)
https://github.com/pedropro/TACO  (licença das imagens: ver o repositório)

O TACO tem fotos de lixo de verdade (rua, praia, chão) com uma caixa marcada
à mão em cada objeto e o tipo de cada um ("Clear plastic bottle", "Drink can"...).
Usamos para testar o fluxo do projeto (YOLO + ResNet) em cenas reais.

Baixa só as fotos que têm garrafas, latas, vidro, copos, caixinhas e plásticos,
na versão de 640 pixels (~900 fotos, ~70 MB). Não precisa instalar nada.

Uso:  dois cliques em baixar_fotos_taco.bat  (ou: python ferramentas/baixar_fotos_taco.py)
Resultado: pasta dados_taco/ e o arquivo dados_taco.zip
"""
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL_ANOTACOES = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
ALVO = {"Clear plastic bottle", "Other plastic bottle", "Glass bottle", "Drink can", "Food Can",
        "Drink carton", "Disposable plastic cup", "Plastic film", "Crisp packet",
        "Corrugated carton", "Other carton", "Glass jar"}

PASTA = Path(__file__).resolve().parent.parent / "dados_taco"
PASTA_IMG = PASTA / "imagens"


def baixar(url: str, destino: Path, tentativas=3) -> bool:
    if destino.exists() and destino.stat().st_size > 0:
        return True
    for _ in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(destino, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except Exception:
            continue
    return False


def main():
    PASTA_IMG.mkdir(parents=True, exist_ok=True)
    print("1/3 Baixando a lista de fotos e marcações do TACO...")
    with urllib.request.urlopen(URL_ANOTACOES, timeout=60) as r:
        anot = json.load(r)
    nomes = {c["id"]: c["name"] for c in anot["categories"]}
    tem_alvo = {a["image_id"] for a in anot["annotations"] if nomes[a["category_id"]] in ALVO}
    imagens = [im for im in anot["images"] if im["id"] in tem_alvo]
    ids = {im["id"] for im in imagens}
    # guarda só as marcações das fotos escolhidas
    sub = {"categories": anot["categories"], "images": imagens,
           "annotations": [a for a in anot["annotations"] if a["image_id"] in ids]}
    (PASTA / "annotations.json").write_text(json.dumps(sub), encoding="utf-8")

    print(f"2/3 Baixando {len(imagens)} fotos (pode levar alguns minutos)...")
    tarefas = [(im["flickr_640_url"], PASTA_IMG / f"{im['id']}.jpg") for im in imagens]
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, deu in enumerate(ex.map(lambda t: baixar(*t), tarefas), 1):
            ok += deu
            if i % 50 == 0 or i == len(tarefas):
                print(f"    {i}/{len(tarefas)}  ({ok} ok)")

    print("3/3 Juntando tudo em dados_taco.zip ...")
    zip_path = shutil.make_archive(str(PASTA), "zip", root_dir=PASTA.parent, base_dir=PASTA.name)
    print(f"\nPronto! {ok} de {len(tarefas)} fotos baixadas.\nArquivo: {zip_path}")
    if ok < len(tarefas):
        print("Algumas falharam. Rode de novo: ele só baixa as que faltam.")


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"\n[ERRO] {erro}\nVeja se a internet está ligada e tente de novo.")
        sys.exit(1)

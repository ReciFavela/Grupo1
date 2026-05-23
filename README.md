# Recifavela-Treino

Treinamento de modelos de Inteligência Artificial para o projeto **Recifavela** — Grupo 1.

Projeto voltado para classificação de resíduos recicláveis, com foco inicial na detecção de materiais **PET** e **NOT_PET**, utilizando visão computacional e aprendizado profundo.

## Objetivo

Desenvolver e treinar modelos de classificação de imagens para auxiliar soluções do projeto Recifavela, explorando:

* Classificação binária (PET / NOT_PET)
* Experimentação com datasets sintéticos e reais
* Técnicas de data augmentation para melhorar generalização
* Fine-tuning de modelos pré-treinados
* Avaliação com métricas como:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

## Dataset

Dataset base utilizado:

https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset

Baixar e colocar em:

```bash
data/
```

Além do dataset base, o projeto inclui curadoria e complementação manual de imagens para reduzir confusões entre classes e melhorar robustez do modelo.

## Estrutura do Projeto

```
Recifavela-Treino/
├── data/
│   ├── PET/
│   └── NOT_PET/
├── models/
│   ├── best_model.pth
│   └── checkpoint.pth
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── worker/
│   └── ingestor.py
├── docker-compose.yml
├── setup_bancos.py
├── requirements.txt
├── imag-test/
├── .env
└── README.md
```

## Treinamento

O treinamento utiliza fine-tuning em redes convolucionais, com estratégias como:

* Congelamento e descongelamento progressivo de camadas
* Augmentations para melhorar generalização
* Salvamento do melhor modelo (best_model)
* Validação por métricas e análise de erros

Exemplo de execução:

```bash
python src/train.py
```

## Avaliação

Para avaliar o modelo treinado:

```bash
python src/evaluate.py
```

Exemplo de métricas obtidas em experimentos:

```
Accuracy: 99%+
F1-score PET: ~97%
```

## Tecnologias

* Python
* PyTorch
* Torchvision
* PIL
* Scikit-learn

## Observações

* O dataset não está versionado no repositório por tamanho.
* Parte do projeto envolve experimentação contínua com novos dados para reduzir falsos positivos e falsos negativos.
* O foco é melhorar a identificação de embalagens PET em cenários variados (cores, formatos, fundos e iluminação).

---

## Infraestrutura de Dados

Nesta etapa foi adicionada a infraestrutura de armazenamento para suportar o pipeline de detecção em produção, integrando YOLO ao classificador existente.

### Arquitetura

```
Imagem (câmera ou dataset)
        │
        ▼
   YOLOv8 — detecta regiões de interesse
        │
        ▼
   best_model.pth — classifica cada região (PET / NOT_PET)
        │
        ├──▶ MinIO    — armazena a imagem original
        ├──▶ MongoDB  — grava o evento bruto do frame
        └──▶ InfluxDB — grava métricas agregadas por turno
```

### Serviços

| Serviço   | Função                                         | Porta |
|-----------|------------------------------------------------|-------|
| MongoDB   | Eventos brutos por frame (um documento/frame)  | 27017 |
| InfluxDB  | Métricas agregadas por classe e turno          | 8086  |
| MinIO     | Armazenamento das imagens (path salvo no Mongo)| 9000  |

### Subindo os serviços

```bash
docker compose up -d
```

Verificar se estão rodando:

```bash
docker compose ps
```

### Configurando os bancos

Execute uma vez antes de rodar o worker:

```bash
pip install -r requirements.txt
python setup_bancos.py
```

Isso cria:
* Collection `frames` no MongoDB com índices em `timestamp`, `camera_id` e `turno`
* Bucket `deteccoes` no InfluxDB
* Bucket `frames` no MinIO

### Worker de Ingestão

O worker processa imagens de uma pasta, detecta objetos com YOLO, classifica com `best_model.pth` e grava nos três bancos.

```bash
python ingestor.py --imagens data/PET --camera cam_01 --turno manha
```

Parâmetros:

| Parâmetro  | Descrição                        | Padrão  |
|------------|----------------------------------|---------|
| `--imagens`| Pasta com imagens a processar    | —       |
| `--camera` | ID da câmera                     | cam_01  |
| `--turno`  | Turno: manha, tarde ou noite     | manha   |

### Estrutura do documento MongoDB (por frame)

```json
{
  "frame_id": "cam_01_a3f8b21c",
  "timestamp": "2026-05-23T08:30:00Z",
  "camera_id": "cam_01",
  "turno": "manha",
  "imagem_path": "frames/cam_01/cam_01_a3f8b21c.jpg",
  "deteccoes": [
    {
      "classe": "PET",
      "confianca": 0.96,
      "conf_yolo": 0.88,
      "bbox": [120, 45, 300, 280]
    }
  ],
  "total_pet": 1,
  "total_not_pet": 0
}
```

### Métricas no InfluxDB

```
Measurement : deteccao_pet
Tags        : camera_id, turno, classe
Fields      : contagem, confianca_media
```

---

## Imagens do último treinamento e do melhor modelo

<img width="1920" height="1032" alt="Captura de tela 2026-04-26 040404" src="https://github.com/user-attachments/assets/0127eaad-50f8-41ff-9744-5296e31332e5" />
<img width="1920" height="1032" alt="Captura de tela 2026-04-26 040327" src="https://github.com/user-attachments/assets/a7b3d936-7a3b-4c00-bb65-d9cdd4047626" />

## Projeto

Projeto desenvolvido no contexto do **Recifavela**, Grupo 1.
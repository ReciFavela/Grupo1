# Status Report 

**Grupo:** Grupo 1 - PET

**Data:** 18/09/2026

---

## Integrantes

| Nome | RA |
| :--- | :---|
| **João Victor Pessoa de Lima dos Anjos** | 23.01078-9 |
| **Vinicius Takashi Nakatsui** | 23.01037-0 |
| **Leonardo Souza Olivieri** | 23.01512-8 |
| **Ilan Hameiry** | 23.00981-0 |
| **Arthur Silva Correia** | 23.00877-6

---

## 1. Status Atual do Projeto

> **Status:** `Em andamento`

---

## 2. Acurácia / Precisão do Modelo

* **Treino:**
  * **Acurácia:** `99,23%`
  * **Precisão Ponderada:** `99,24%`
  * **Recall Ponderado:** `99,23%`
* **Avaliação:**
  * **Confiança Média PET:** `94,8%`
  * **Confiança Média NOT_PET:** `98,3%`

## 3. Base de Dados Utilizada

* ***Bottles Synthetic Images:*** [Acessar Dataset no Kaggle](https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset)
* ***RealWaste Image Classification:*** [Acessar Dataset no Kaggle](https://www.kaggle.com/datasets/joebeachcapital/realwaste)

---

## 4. Quantidade de Imagens Utilizadas

* **Treinamento:** **14.366 imagens** (*Divisão:* 80% treino / 20% teste)
* **Avaliação:** **71.779 imagens**

---

## 5. Resumo das Tecnologias Utilizadas

### Linguagem de Programação

* Python
  * Bibliotecas Externas
    * `torch`
    * `torchvision`
    * `tqdm`
    * `sklearn`
    * `PIL` 
    * `ultralytics` (ResNet18 e YOLOv8)
    * `pymongo`
    * `influxdb-client`
    * `minio`
      
### Banco de Dados

* MongoDB
* InfluxDB
* MinIO

### Containerização
* Docker

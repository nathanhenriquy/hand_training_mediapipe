# Reconhecimento de Gestos com as Mãos em Tempo Real
---

## 📂 Estrutura do Projeto

Aqui está uma descrição de cada arquivo e diretório principal do projeto:

-   `📁 dataset/`: Diretório que armazena as imagens de cada gesto utilizadas para o treinamento. Cada subpasta dentro de `dataset` representa uma classe de gesto (ex: `A/`, `B/`, `C/`).
-   `📄 fazer_dataset.ipynb`: Um Jupyter Notebook utilizado para coletar as imagens que compõem o dataset. Ele auxilia na captura de imagens da webcam e na organização das mesmas nas pastas corretas.
-   `📄 treino.py`: Script Python responsável por processar os dados do `dataset`, treinar o modelo de classificação (usando `RandomForestClassifier`) e salvar o modelo treinado no arquivo `model.p`.
-   `📄 inferencia.py`: O script principal da aplicação. Ele carrega o modelo treinado (`model.p`), inicia a webcam e realiza a detecção e classificação dos gestos em tempo real, exibindo o resultado na tela.
-   `📄 model.p`: Arquivo binário que contém o modelo de Machine Learning já treinado. É gerado pelo `treino.py` e utilizado pelo `inferencia.py`.

---

## 🛠️ Tecnologias Utilizadas

-   **Python 3.x**
-   **OpenCV (`opencv-python`)**: Para captura e manipulação de vídeo da webcam.
-   **MediaPipe**: Para a detecção dos landmarks da mão em tempo real.
-   **Scikit-learn**: Para o treinamento do modelo de classificação.
-   **NumPy**: Para manipulação eficiente de arrays numéricos.
-   **Jupyter Notebook**: Para a etapa interativa de coleta de dados.

---


## Video demonstração
- https://drive.google.com/file/d/1QGOFt2yAIscC2j9ZZX_7zY2wjTsZETwa/view?usp=sharing

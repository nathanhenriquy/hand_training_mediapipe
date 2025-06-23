import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tentei realizar o trabalho seguindo o exemplo do professor com redes neurais, porém não deu certo, pois após treinamento a predição mostra
# apenas um resultado, essa abordagem a seguir pega os pontos do midiapipe das mãos e treina por eles.

# Dados

detectar_maos = mp.solutions.hands
maos = detectar_maos.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# caminho para o dataset, dentro dele tem pastas a,b,c,d.... com imangens referentes dentro
# imagens ddentro tem tirados por mim com o .py, e imagens de um dataset do kaggle
dataset = './dataset'
# armazena os pontos de referencia da coordenadas da mao
data = []
# armazena os nomes corretos correspondente a "data", "data" guarda numeros e labels a resposta
labels = []


# processar imagens
for pasta in os.listdir(dataset):
    
    if pasta.startswith('.'):
        continue
    
    # printa o nome da pasta q esta processadndo
    print(f"{pasta}")

    # itera sobre cada arquivo de imagem dentro da pasta 
    for img_path in os.listdir(os.path.join(dataset, pasta)):
        # guarda temporariamente x e y da mao na imagem atual.
        data_aux = []
        
        # Constrói o caminho completo para a imagem.
        full_img_path = os.path.join(dataset, pasta, img_path)
        # lea imagem do arquivo usando OpenCV
        img = cv2.imread(full_img_path)
        # converte a imagem para RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # processa a imagem RGB com detector de mao
        resultado = maos.process(img_rgb)
        
        # verifica se a mao foi detectada
        if resultado.multi_hand_landmarks:
            # pega os pontos
            hand_landmarks = resultado.multi_hand_landmarks[0]
            
            # pega as coordenadas do ponto 0 (o pulso)
            base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y

            # e pega o restandte dos pontos
            for landmark in hand_landmarks.landmark:
                # calcula o x e y  de cada ponto subtraindo as coordenadas do pulso.
                data_aux.append(landmark.x - base_x)
                data_aux.append(landmark.y - base_y)

            # para normalizar a escala
            max_dist = 0
            for i in range(len(hand_landmarks.landmark)):
                # calcula a distancia  do ponto atual ao pulso.
                dist = np.sqrt((hand_landmarks.landmark[i].x - base_x)**2 + (hand_landmarks.landmark[i].y - base_y)**2)
                # atualiza a distancia max se a distancia atual for maior.
                if dist > max_dist:
                    max_dist = dist
            
            # n deixa dividir por zero caso a mao n tenha sido detectada
            if max_dist > 0:
                data_aux_scaled = [val / max_dist for val in data_aux]
                
                # coloca os datos normalizados em "data"
                data.append(data_aux_scaled)
                # e guarda o gabarito do dados em "labels"
                labels.append(pasta)


print("Process pronto") # quando terminar as pastas

# Treinamento

# pego os modelos 'data' e 'labels' e converto parapara arrays NumPy
data = np.asarray(data)
labels = np.asarray(labels)

# dividir os dados para treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# inicia o treinamento
model = RandomForestClassifier()
# treina o modelo usando os dados 'x_train' são as features e  'y_train' são os rótulos
model.fit(x_train, y_train)

# teste
y_predict = model.predict(x_test)
# calcula a acuracia 
score = accuracy_score(y_test, y_predict)
print(f'Acuracia final: {score * 100:.2f}%')

# salva modelo
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("\ndados processados e treinados")
import pickle 
import cv2      
import mediapipe as mp 
import numpy as np      
import os 

model = None

# parte da inferencia feita junto com IA, para realizar a predicao de forma dinamica, mudando o sinal ja mostra o resultado

# carrega o modelo

with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# inicia a camera
camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands # detectar as maos
mp_drawing = mp.solutions.drawing_utils # desenhar os pontos
mp_drawing_styles = mp.solutions.drawing_styles # estilo do desenho

# detector de maos
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# loop para inferencia
while True:
    # le um frame da camera. 'ret' é um booleano 
    ret, frame = camera.read()
    if not ret:
        print("Encerrando detector")
        break
    
    # pega as dimensoes
    H, W, _ = frame.shape
    # igual no treinamento, converte para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # pega o frame para detectar as maos
    results = hands.process(frame_rgb)

    # caso a mao for detectada
    if results.multi_hand_landmarks:
        # pega os landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Desenha os pontos 
        mp_drawing.draw_landmarks(
            frame,                   # Imagem onde desenhar.
            hand_landmarks,          # Coordenadas dos pontos.
            mp_hands.HAND_CONNECTIONS, # Conexões entre os pontos (esqueleto da mão).
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # normalizacao dos dados igual ao treino
        data_aux = []
        base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y

        # calcula as coordenadas 
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - base_x)
            data_aux.append(landmark.y - base_y)
        
        
        max_dist = max(np.sqrt((lm.x - base_x)**2 + (lm.y - base_y)**2) for lm in hand_landmarks.landmark)
        
        # Garante que a distância máxima não seja zero para evitar divisão por zero.
        if max_dist > 0:
            
            data_aux_scaled = [val / max_dist for val in data_aux]
            input_data = np.asarray(data_aux_scaled).reshape(1, -1)

            # predicao
            predicted_char_result = model.predict(input_data)[0]
            
            #obtor confianca
            prediction_probs = model.predict_proba(input_data)[0]
            
            
            # model.classes_ contém a lista de todas as classes que o modelo conhece.
            class_index = list(model.classes_).index(predicted_char_result)
            confidence = prediction_probs[class_index]
            
            predicted_character = predicted_char_result

            # desenha o resultado na tela
            x_coords = [lm.x * W for lm in hand_landmarks.landmark]
            y_coords = [lm.y * H for lm in hand_landmarks.landmark]
            x1, y1 = int(min(x_coords)) - 10, int(min(y_coords)) - 10

            # texto de resultado que mostra na tela
            text_to_show = f'{predicted_character} ({confidence*100:.1f}%)'
            
            # Desenha um retângulo preto como fundo para o texto, para melhor legibilidade.
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(text_to_show) * 15, y1), (0, 0, 0), -1)
            # Escreve o texto da predição no frame.
            cv2.putText(frame, text_to_show, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # mostrar o resultado
    cv2.imshow('Inferencia de LIBRAS - Pressione "q" para sair', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

camera.release()
# fecha as janelas abertas
cv2.destroyAllWindows()

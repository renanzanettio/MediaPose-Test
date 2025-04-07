import cv2 # Importa a biblioteca OpenCV para processamento dos Vídeos/Webcam
import mediapipe as mp # Importa a biblioteca MediaPipe para a detecção de pose humana, define variável mp para ela
import time # Importa a bibliotecca time para medir o tempo e calcular o FPS


mpDraw = mp.solutions.drawing_utils # Variável do modulo para desenhar os pontos e conexões no corpo from MediaPipe
mpPose = mp.solutions.pose # Variável do módulo de detecção de pose from MediaPipe
pose = mpPose.Pose() # Cria um objeto para processar as poses corporais em imagens

cap = cv2.VideoCapture("Videos/1.mp4") # Define o que vai ser lido em vídeo e gaurdado como cap
pTime = 0 # Variável para armazenar o tempo do frame anterior (usado no cálculo de FPS)

while True: # Loop principal, executa infinitamente
    success, img = cap.read() # Lê um frame do vídeo; "sucess" indica se a leitura foi bem-sucedida
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converte a imagem de BGR para RGB (MediaPipe usa RGB)
    results = pose.process(imgRGB) # Processa a imagem RGB para detectar a pose corporal

    #print(results.pose_landmarks)  # Modo simples de mostrar as coordenadas dos pontos do corpo (não mostra o ID de cada um)

    if results.pose_landmarks: # Se alguma pose for detectada
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # Desenha os pontos e conexões da pose na imagem original
        for id, lm in enumerate(results.pose_landmarks.landmark): # Percorre cada ponto detectado
            h, w, c = img.shape # Obtém altura, largura e canais da imagem; canais de imagem (3= RGB; 1= P&B)
            print(id, lm) # Exibe o índice do ponto e suas coordenadas normalizadas
            cx, cy = int(lm.x * w) , int(lm.y * h) # Converte as coordenadas normalizadas para coordenadas em pixels
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) # Desenha um círculo azul preenchido no ponto correspondente da imagem (onde, posição, tamanho, cor, preenchido)


    cTime = time.time() # Armazena o tempo atual
    fps = 1 / (cTime - pTime) # Calcula o FPS com base no tempo entre os frames
    pTime = cTime # Atualiza o tempo anterior

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) # Escreve o valor de FPS na tela (onde, texto, posição, fonte, tamanho, cor, espessura)
    cv2.imshow("Image", img) # Exibe a imagem com as anotações (ponto, conexões, FPS)

    cv2.waitKey(1) # Aguarda 1 milissegundo entre os frames (serve para a janela funcionar corretamente)

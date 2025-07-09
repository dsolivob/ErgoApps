import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Pose y la cámara web
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configuración para la detección de poses
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detectar_mala_postura(landmarks):
    """
    Detecta si la postura es incorrecta evaluando la alineación de los hombros y la posición de la cabeza.
    """
    # Obtiene los puntos clave de los hombros y la cabeza
    hombro_izq = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hombro_der = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    oreja_izq = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    oreja_der = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    
    # Verifica si los hombros están alineados
    alineacion_hombros = abs(hombro_izq.y - hombro_der.y)
    
    # Verifica si la cabeza está torcida (ángulo entre las orejas)
    angulo_cabeza = abs(oreja_izq.y - oreja_der.y)

    # Definir límites para la alineación (ajustar según sea necesario)
    limite_hombros = 0.05  # Límite de tolerancia para alineación de hombros
    limite_cabeza = 0.05   # Límite de tolerancia para inclinación de cabeza

    # Detecta mala postura si los hombros no están alineados o la cabeza está torcida
    mala_postura = (alineacion_hombros > limite_hombros) or (angulo_cabeza > limite_cabeza)
    
    return mala_postura

# Captura el video de la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convierte el frame a RGB (requerido por MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesa la imagen para detectar la pose
    resultado = pose.process(frame_rgb)
    
    # Si se detectan puntos de referencia
    if resultado.pose_landmarks:
        # Dibuja los puntos de referencia en la imagen
        mp_drawing.draw_landmarks(frame, resultado.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Verifica si hay mala postura
        mala_postura = detectar_mala_postura(resultado.pose_landmarks.landmark)
        
        # Muestra un mensaje si se detecta mala postura
        if mala_postura:
            cv2.putText(frame, "Mala Postura Detectada!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Postura Correcta", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Muestra el frame con anotaciones
    cv2.imshow("Detector de Mala Postura", frame)
    
    # Salir del loop con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()

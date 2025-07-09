import cv2
import mediapipe as mp
import numpy as np
import math

def calcular_angulo(punto1, punto_medio, punto2):
    """Calcula el ángulo entre tres puntos."""
    vector1 = np.array(punto1) - np.array(punto_medio)
    vector2 = np.array(punto2) - np.array(punto_medio)

    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0  # Evitar división por cero

    cos_angulo = np.dot(vector1, vector2) / (norm_vector1 * norm_vector2)
    angulo_rad = np.arccos(np.clip(cos_angulo, -1.0, 1.0))
    angulo_deg = np.degrees(angulo_rad)
    return angulo_deg

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando fotograma de cámara vacío.")
        continue

    # Invertir la imagen horizontalmente para una vista de espejo (opcional)
    image = cv2.flip(image, 1)

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe Pose
    results = pose.process(image_rgb)

    # Dibujar los puntos clave y las conexiones en la imagen
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            landmarks = results.pose_landmarks.landmark
            h, w, c = image.shape

            # **Medición del ángulo del codo izquierdo**
            hombro_izquierdo = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)]
            codo_izquierdo = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h)]
            muneca_izquierda = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h)]

            angulo_codo_izquierdo = calcular_angulo(hombro_izquierdo, codo_izquierdo, muneca_izquierda)

            # Mostrar el ángulo sobre el punto del codo izquierdo
            cv2.putText(image, f'{angulo_codo_izquierdo:.2f}',
                        (codo_izquierdo[0], codo_izquierdo[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # **Medición del ángulo del codo derecho**
            hombro_derecho = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)]
            codo_derecho = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)]
            muneca_derecha = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)]

            angulo_codo_derecho = calcular_angulo(hombro_derecho, codo_derecho, muneca_derecha)

            # Mostrar el ángulo sobre el punto del codo derecho
            cv2.putText(image, f'{angulo_codo_derecho:.2f}',
                        (codo_derecho[0], codo_derecho[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            pass

    cv2.imshow('Medición de Ángulos del Codo', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
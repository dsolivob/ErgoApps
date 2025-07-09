import cv2
import mediapipe as mp

# Inicializar mediapipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Función para contar los dedos levantados
def contar_dedos(hand_landmarks, hand_type):
    dedos_levantados = 0
    dedos_ids = [4, 8, 12, 16, 20]  # Pulgar, índice, medio, anular y meñique
    
    # Comprobación para el pulgar (cambia según mano izquierda o derecha)
    if hand_type == "Right":
        if hand_landmarks[dedos_ids[0]].x < hand_landmarks[dedos_ids[0] - 1].x:
            dedos_levantados += 1
    else:
        if hand_landmarks[dedos_ids[0]].x > hand_landmarks[dedos_ids[0] - 1].x:
            dedos_levantados += 1

    # Comprobación para los otros dedos
    for id in range(1, 5):
        if hand_landmarks[dedos_ids[id]].y < hand_landmarks[dedos_ids[id] - 2].y:
            dedos_levantados += 1

    return dedos_levantados

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB para mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Variables para contar dedos levantados en ambas manos
    dedos_izquierda = 0
    dedos_derecha = 0

    # Procesar cada mano detectada
    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_type = handedness.classification[0].label  # "Left" o "Right"

            # Contar dedos para cada mano
            dedos = contar_dedos(hand_landmarks.landmark, hand_type)
            if hand_type == "Left":
                dedos_izquierda = dedos
            else:
                dedos_derecha = dedos

            # Dibujar puntos de referencia de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar conteo de dedos en pantalla
    cv2.putText(frame, f"Dedos Mano Izquierda: {dedos_izquierda}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Dedos Mano Derecha: {dedos_derecha}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow("Deteccion de Dedos Levantados", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageTk
import sys # Importar el módulo sys para sys.exit()

# --- Funciones de procesamiento de MediaPipe ---
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

# --- Variables globales para la GUI ---
cap = None
processing_active = False
current_source_type = None

# --- Funciones para la GUI ---
def select_image():
    global cap, processing_active, current_source_type
    if processing_active:
        stop_processing_internal() # Usar la función interna para detener sin cerrar
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        current_source_type = "image"
        process_image(file_path)

def select_video():
    global cap, processing_active, current_source_type
    if processing_active:
        stop_processing_internal() # Usar la función interna para detener sin cerrar

    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    if file_path:
        current_source_type = "video"
        start_video_processing(file_path)

def start_webcam():
    global cap, processing_active, current_source_type
    if processing_active:
        stop_processing_internal() # Usar la función interna para detener sin cerrar
    
    # Try different camera indices to find an available webcam
    found_webcam = False
    for i in range(3): # Check indices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found_webcam = True
            break
        cap.release()

    if found_webcam:
        current_source_type = "webcam"
        processing_active = True
        update_frame()
    else:
        messagebox.showerror("Error", "No se pudo acceder a la cámara web. Asegúrate de que no esté en uso por otra aplicación.")

def stop_processing_internal():
    """Detiene el procesamiento de la cámara/video sin cerrar la aplicación."""
    global cap, processing_active
    processing_active = False
    if cap and cap.isOpened():
        cap.release()
    canvas.delete("all")
    label_info.config(text="Selecciona una opción para empezar a procesar.")

def stop_and_exit():
    """Detiene el procesamiento y cierra la aplicación."""
    global cap, processing_active
    processing_active = False
    if cap and cap.isOpened():
        cap.release()
    pose.close() # Es buena práctica cerrar la instancia de MediaPipe
    window.destroy() # Destruye la ventana de Tkinter
    sys.exit() # Sale del programa Python

def process_image(image_path):
    global processing_active
    processing_active = True
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "No se pudo cargar la imagen.")
        processing_active = False
        return

    display_frame(image)
    processing_active = False # For static image, processing finishes after displaying

def start_video_processing(video_path):
    global cap, processing_active
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"No se pudo abrir el video: {video_path}")
        return

    processing_active = True
    update_frame()

def update_frame():
    global cap, processing_active
    if not processing_active:
        return

    if cap and cap.isOpened():
        success, frame = cap.read()
        if success:
            display_frame(frame)
            window.after(10, update_frame)  # Update every 10ms
        else:
            stop_processing_internal() # Usar la función interna al finalizar el video
            messagebox.showinfo("Fin de procesamiento", "El video ha terminado o hubo un error al leer el fotograma.")
    else:
        stop_processing_internal() # Usar la función interna si la cámara/video se cierra inesperadamente

def display_frame(image):
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

            def obtener_coordenadas(landmark):
                return [int(landmarks[landmark.value].x * w), int(landmarks[landmark.value].y * h)]

            # Puntos clave para la espalda y el cuello (vista lateral)
            hombro = obtener_coordenadas(mp_pose.PoseLandmark.LEFT_SHOULDER) # Asumiendo vista lateral izquierda
            codo = obtener_coordenadas(mp_pose.PoseLandmark.LEFT_ELBOW)
            muneca = obtener_coordenadas(mp_pose.PoseLandmark.LEFT_WRIST)
            cadera = obtener_coordenadas(mp_pose.PoseLandmark.LEFT_HIP)
            rodilla = obtener_coordenadas(mp_pose.PoseLandmark.LEFT_KNEE)
            tobillo = obtener_coordenadas(mp_pose.PoseLandmark.LEFT_ANKLE)
            nariz = obtener_coordenadas(mp_pose.PoseLandmark.NOSE)

            # Ángulo de flexión del codo
            angulo_codo = calcular_angulo(hombro, codo, muneca)
            cv2.putText(image, f'Codo: {angulo_codo:.2f}', (codo[0] + 10, codo[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Ángulo de flexión de la rodilla
            angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
            cv2.putText(image, f'Rodilla: {angulo_rodilla:.2f}', (rodilla[0] + 10, rodilla[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Ángulo de inclinación de la espalda (aproximado)
            vector_cadera_hombro = np.array(hombro) - np.array(cadera)
            vector_horizontal = np.array([1, 0]) # Vector horizontal hacia la derecha

            norm_vector_cadera_hombro = np.linalg.norm(vector_cadera_hombro)
            norm_vector_horizontal = np.linalg.norm(vector_horizontal)

            angulo_espalda_deg = 0
            if norm_vector_cadera_hombro > 0:
                cos_angulo_espalda = np.dot(vector_cadera_hombro, vector_horizontal) / (norm_vector_cadera_hombro * norm_vector_horizontal)
                angulo_espalda_rad = np.arccos(np.clip(cos_angulo_espalda, -1.0, 1.0))
                angulo_espalda_deg = np.degrees(angulo_espalda_rad)
                if hombro[1] > cadera[1]: # Hombro más abajo que la cadera (inclinación hacia adelante)
                    angulo_espalda_deg = -angulo_espalda_deg
                
            cv2.putText(image, f'Espalda: {angulo_espalda_deg:.2f}', (cadera[0] - 30, cadera[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA) # Naranja

            # Ángulo de flexión del cuello (aproximado)
            vector_hombro_nariz = np.array(nariz) - np.array(hombro)
            vector_vertical = np.array([0, -1]) # Vector vertical hacia arriba

            norm_vector_hombro_nariz = np.linalg.norm(vector_hombro_nariz)
            norm_vector_vertical = np.linalg.norm(vector_vertical)

            angulo_cuello_deg = 0
            if norm_vector_hombro_nariz > 0:
                cos_angulo_cuello = np.dot(vector_hombro_nariz, vector_vertical) / (norm_vector_hombro_nariz * norm_vector_vertical)
                angulo_cuello_rad = np.arccos(np.clip(cos_angulo_cuello, -1.0, 1.0))
                angulo_cuello_deg = np.degrees(angulo_cuello_rad)

            cv2.putText(image, f'Cuello: {angulo_cuello_deg:.2f}', (hombro[0] + 10, hombro[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA) # Amarillo

            # Update info label
            label_info.config(text=f"Codo: {angulo_codo:.2f}° | Rodilla: {angulo_rodilla:.2f}° | Espalda: {angulo_espalda_deg:.2f}° | Cuello: {angulo_cuello_deg:.2f}°")

        except Exception as e:
            label_info.config(text=f"Error en el cálculo de ángulos: {e}")
            pass
    else:
        label_info.config(text="No se detectaron puntos clave de pose.")
        
    # Redimensionar la imagen para que quepa en el canvas
    img_height, img_width, _ = image.shape
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Calculate scale factor to fit image within canvas while maintaining aspect ratio
    scale_w = canvas_width / img_width
    scale_h = canvas_height / img_height
    scale = min(scale_w, scale_h)

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    if new_width > 0 and new_height > 0:
        image = cv2.resize(image, (new_width, new_height))

    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    canvas.create_image(canvas_width/2, canvas_height/2, anchor=tk.CENTER, image=img_tk)
    canvas.image = img_tk # Keep a reference!

# --- Configuración de la Ventana Principal ---
window = tk.Tk()
window.title("ergoapp")
window.geometry("1000x800")

# --- Controles de la Interfaz ---
control_frame = tk.Frame(window, bd=2, relief=tk.RAISED, padx=10, pady=10)
control_frame.pack(side=tk.TOP, fill=tk.X)

btn_image = tk.Button(control_frame, text="Procesar Imagen", command=select_image)
btn_image.pack(side=tk.LEFT, padx=5, pady=5)

btn_video = tk.Button(control_frame, text="Procesar Video", command=select_video)
btn_video.pack(side=tk.LEFT, padx=5, pady=5)

btn_webcam = tk.Button(control_frame, text="Cámara Web", command=start_webcam)
btn_webcam.pack(side=tk.LEFT, padx=5, pady=5)

# Botón "Detener y Salir" con estilo y posición
btn_stop = tk.Button(
    control_frame,
    text="Detener y Salir",
    command=stop_and_exit,
    bg="red",          # Fondo rojo
    fg="white",        # Letra blanca
    font=("Helvetica", 10, "bold") # Negrilla y tamaño 10
)
btn_stop.pack(side=tk.RIGHT, padx=5, pady=5) # Posicionado a la derecha

# --- Área de Visualización ---
canvas = tk.Canvas(window, bg="black", width=900, height=600)
canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# --- Etiqueta de Información ---
label_info = tk.Label(window, text="Selecciona una opción para empezar a procesar.", font=("Helvetica", 12))
label_info.pack(side=tk.BOTTOM, pady=10)

# --- Ejecutar la Aplicación ---
# Asignar la función de cierre a la acción de cerrar la ventana
window.protocol("WM_DELETE_WINDOW", stop_and_exit)
window.mainloop()
# 🧠ErgoApps
Aplicaciones inteligentes para ergonomía y bienestar humano. Usa IA, visión artificial y datos biométricos para optimizar la interacción hombre-máquina en entornos laborales, clínicos y educativos.
---

# 🤸‍♀️ ErgoApp: Analizador de Postura en Tiempo Real

**ErgoApp** es una aplicación de escritorio desarrollada en Python que utiliza visión por computador para analizar la postura corporal en imágenes, videos o desde una cámara web en tiempo real. La aplicación calcula ángulos clave del cuerpo para proporcionar un análisis ergonómico personalizado.

---

## 🧠 Componentes Tecnológicos

Este proyecto integra un framework de interfaz gráfica con varias librerías especializadas en visión por computador y cálculo numérico.

### 🏗️ Framework

- **Tkinter**: Framework principal para la interfaz gráfica de usuario (GUI). Administra la ventana principal, botones, lienzo de visualización y el bucle de eventos que responde a las interacciones del usuario.

### 🧰 Librerías Utilizadas

- **OpenCV (`cv2`)**  
  Procesamiento de imágenes y video:
  - Lectura y escritura de archivos
  - Captura de cámara web
  - Manipulación de imagen (color, volteo)
  - Dibujo de texto y formas

- **MediaPipe (`mediapipe`)**  
  Motor de IA para análisis corporal:
  - Detección de 33 puntos clave del cuerpo humano
  - Seguimiento y visualización del esqueleto
  - Extracción de coordenadas para análisis

- **NumPy (`numpy`)**  
  Cálculo numérico:
  - Conversión de coordenadas en vectores
  - Cálculos trigonométricos para ángulos articulares (codo, rodilla, espalda)

- **Pillow (`PIL`)**  
  Conector entre OpenCV y Tkinter:
  - Conversión de fotogramas a formato compatible con la GUI

- **Sys**  
  Librería estándar para interacción con el sistema:
  - Cierre limpio y controlado de la aplicación

---

## 📊 Fuente de Datos (Dataset)

A diferencia de muchos proyectos de IA, **ErgoApp** no se entrena con un dataset estático. El análisis se realiza en tiempo real con datos proporcionados por el usuario.

### Fuentes aceptadas:

- **Imágenes**: `.jpg`, `.png`, etc.
- **Videos**: `.mp4`, `.avi`, etc.
- **Transmisión en vivo**: Captura directa desde una cámara web conectada

---

¿Quieres que te ayude a agregar una sección de instalación, uso o ejemplos visuales para completar el README? También podemos incluir badges, GIFs o enlaces a notebooks de demostración. 

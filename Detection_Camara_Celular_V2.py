from ultralytics import YOLO
import cv2

# Cargamos el modelo YOLO entrenado
model = YOLO("best.pt")

# Cámara web
# cap = cv2.VideoCapture(0)

# DroidCam
cap = cv2.VideoCapture("http://192.168.1.16:4747/video")

# Verificar conexión
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while cap.isOpened():

    # Leemos el frame del video
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame")
        break

    # Realizamos la inferencia de YOLO sobre el frame
    results = model(frame)

    # Dibujamos las detecciones
    annotated_frame = results[0].plot()

    # Mostramos el resultado
    cv2.imshow(
        "YOLO PPE Detection",
        annotated_frame
    )

    # Presionar ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
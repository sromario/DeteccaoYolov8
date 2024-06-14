from ultralytics import YOLO
import cv2

# Carregar o modelo treinado
model = YOLO('runs/detect/train/weights/best.pt')

# Função para realizar a inferência em tempo real na webcam
def predict_webcam(model, conf=0.7, iou=0.7, imgsz=320, min_confidence=0.25):
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture('coca.mp4')

    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    while True:
        # Captura frame por frame
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar o frame do vídeo")
            break

        # Redimensiona o frame para o tamanho desejado
        frame_resized = cv2.resize(frame, (imgsz, imgsz))

        # Realiza a inferência no frame capturado
        results = model.predict(
            source=frame_resized,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            stream=True  # Stream permite a iteração sobre os resultados em tempo real
        )

        # Iterar sobre os resultados do gerador
        for result in results:
            # Desenhar as caixas delimitadoras no frame
            if result.boxes:
                for box in result.boxes:
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    class_name = model.names[class_id]

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Desenhar a caixa e o texto no frame original
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar o frame com as detecções
            cv2.imshow('Video Detection', frame)

        # Espera por uma tecla ser pressionada para sair do loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Libera a captura de vídeo e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

# Chama a função para realizar a inferência no vídeo
predict_webcam(model, conf=0.7, iou=0.45, imgsz=320)

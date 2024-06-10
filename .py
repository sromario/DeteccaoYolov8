from ultralytics import YOLO
import cv2

# Carregar o modelo treinado
model = YOLO('runs/detect/train2/weights/best.pt')

# Função para salvar os resultados da validação do modelo
def evaluate_model(model, project='runs/detect', imgsz=640, batch=16, conf=0.001, iou=0.7, save_json=False, save_hybrid=False, split='test'):
    results = model.val(
        project=project,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,  # Non-Maximum Suppression (NMS)
        save_json=save_json,  # Save to JSON {image_id, cls, bbox, conf} of each image in dataset
        save_hybrid=save_hybrid,  # Bounding box labels + inference on the output image
        split=split  # train, val or test
    )
    return results.box

def predict_image(model, image_path, conf=0.25, iou=0.7, imgsz=640, show=False, save=True, save_txt=True, save_conf=True, save_crop=True, stream=False):
    # Ler a imagem usando OpenCV
    image = cv2.imread(image_path)
    
    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return []

    # Realizar a inferência na imagem
    results_img = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        show=show,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        stream=stream
    )
    
    # Processar os resultados para extrair as classes e bounding boxes
    detections = results_img[0].boxes if results_img else None
    if detections:
        classes = detections.cls.cpu().numpy()  # Obter os índices das classes
        labels = [model.names[int(cls)] for cls in classes]  # Converter os índices das classes para nomes das classes
    else:
        labels = []
    
    print(f"Image: {image_path}, Detected classes: {labels}")

    # Exibir a imagem com as detecções usando OpenCV
    if show:
        result_image = results_img[0].plot()  # Isso plota as caixas delimitadoras na imagem
        cv2.imshow("Predictions", result_image)
        
        # Esperar por qualquer evento de tecla (0 indica espera indefinida)
        cv2.waitKey(0)
        
        # Fechar a janela após qualquer tecla ser pressionada
        cv2.destroyAllWindows()

    return labels
image_path = '07.jpeg'  # Atualize com o caminho para sua imagem

# Mostrar os resultados
model.predict(source=image_path, conf=0.25, iou=0.45, show=True)  # Ajuste os valores de confiança e IoU conforme necessário
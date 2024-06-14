from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Carregar o modelo treinado
model = YOLO('runs/detect/train/weights/best.pt')

# Função para fazer a inferência em uma imagem
def predict_image(model, image_path, conf=0.7, iou=0.5, imgsz=640):
    # Ler a imagem usando OpenCV
    image = cv2.imread(image_path)
    
    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None

    # Realizar a inferência na imagem
    results_img = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        stream=False
    )
    
    return results_img

# Função para exibir a imagem com as caixas delimitadoras
def display_image_with_boxes(image_path, results_img, ax):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)

    detections = results_img[0].boxes if results_img else None
    if detections:
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Converte o tensor para float para formatar corretamente
            ax.text(x1, y1 - 10, f'{model.names[int(box.cls)]}: {float(box.conf):.2f}', color='red', fontsize=12, weight='bold')

    ax.axis('off')

# Obter a lista de imagens na pasta 'teste'
test_folder = 'teste'
test_images = [os.path.join(test_folder, img) for img in os.listdir(test_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Configurar o plot
fig, axes = plt.subplots(1, len(test_images), figsize=(15, 10))

# Realizar a inferência e exibir resultados para cada imagem
for ax, image_path in zip(axes, test_images):
    results = predict_image(model, image_path, conf=0.7, iou=0.5)
    display_image_with_boxes(image_path, results, ax)

plt.show()

from ultralytics import YOLO, settings
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

print(settings)

#baixar modelo pronto
model = YOLO('yolov8n') # ou arquitetura customizada

# treinar o modelo
# Reduzindo o tamanho do lote, o tamanho das imagens de entrada e o número de épocas
model.train(data="custom_dataset.yaml",
            epochs=15,  # Reduzindo pela metade o número de épocas
            batch=8,    # Reduzindo pela metade o tamanho do lote
            imgsz=416,  # Reduzindo o tamanho das imagens de entrada
            workers=4,  # Reduzindo o número de workers
            # Desligando algumas augmentations ou reduzindo sua intensidade
            degrees=0.1,
            hsv_s=0.1,
            hsv_v=0.1,
            scale=0.2,
            fliplr=0.2
            )

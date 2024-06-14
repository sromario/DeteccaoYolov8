from ultralytics import YOLO, settings
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS



#baixar modelo pronto
model = YOLO('yolov8n') # ou arquitetura customizada

# treinar o modelo
model.train(data="custom_dataset.yaml",
            epochs=30,  # quantidade de rodadas
            batch=16,   # lote usado por rodada
            imgsz=640,  # redimensionar imgs
            workers=8,  #  número de subprocessos para processar imgs
            # Ajustando as augmentations 
            degrees=0.0,    # rotação
            hsv_s=0.5,      # Saturação HSV padrão
            hsv_v=0.5,      # Valor HSV padrão
            scale=0.5,      # Escala padrão
            fliplr=0.5      # Flip horizontal padrão
            )
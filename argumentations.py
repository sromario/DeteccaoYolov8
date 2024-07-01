import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np

# Definir o pipeline de augmentations com ênfase na forma arredondada da garrafa
augmentation = A.Compose([
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(p=0.2),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RGBShift(p=0.2),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.5),

    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Função para aplicar augmentations e salvar as imagens e anotações
def augment_and_save(image_path, label_path, output_image_dir, output_label_dir, augmentation, num_augments=5):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    with open(label_path, 'r') as file:
        labels = file.readlines()
        
    bboxes = []
    class_labels = []
    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.strip().split())
        # Normalizando as coordenadas das caixas delimitadoras
        x_center /= width
        y_center /= height
        bbox_width /= width
        bbox_height /= height
        bboxes.append([x_center, y_center, bbox_width, bbox_height])
        class_labels.append(int(class_id))
    
    for i in range(num_augments):
        augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']
        
        # Convertendo a imagem de volta para NumPy array
        augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()  # Convertendo para formato (H, W, C)
        augmented_image = (augmented_image * 255).astype(np.uint8)  # Escalando de volta para [0, 255] e convertendo para uint8
        
        output_image_path = os.path.join(output_image_dir, f"{os.path.basename(image_path).split('.')[0]}_aug_{i}.jpg")
        output_label_path = os.path.join(output_label_dir, f"{os.path.basename(label_path).split('.')[0]}_aug_{i}.txt")
        
        cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))  # Salvando a imagem aumentada
        
        with open(output_label_path, 'w') as file:
            for bbox, class_label in zip(augmented_bboxes, augmented_class_labels):
                file.write(f"{class_label} {' '.join(map(str, bbox))}\n")

# Diretórios de entrada e saída
input_image_dir = 'dataset/retornavel/train/images'
input_label_dir = 'dataset/retornavel/train/labels'
output_image_dir = 'dataset/retornavel/train/aug_images'
output_label_dir = 'dataset/retornavel/train/aug_labels'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Aplique augmentations em todas as imagens e suas anotações no diretório
for image_name in os.listdir(input_image_dir):
    image_path = os.path.join(input_image_dir, image_name)
    label_path = os.path.join(input_label_dir, image_name.replace('.jpg', '.txt'))
    
    # Verificar se o arquivo de anotação existe antes de processar
    if os.path.isfile(label_path):
        augment_and_save(image_path, label_path, output_image_dir, output_label_dir, augmentation)
    else:
        print(f"Arquivo de anotação não encontrado para {image_path}")

import cv2
import os
from datetime import datetime

# iniciar camera
cam = cv2.VideoCapture(0)

# local onde imgs será salva, criar caso não exista
save_directory = "dataset/data"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

file_name = "image.jpg"
file_path = os.path.join(save_directory, file_name)


print("Pressione 'o' para tirar foto ou 'x' para fechar.")

image_count = 0  # Contador para o número de imagens salvas

# ler frames e capturar
while True:

    ret, frame = cam.read()

    cv2.imshow('Imagem capturada', frame)

    # Aguarda por uma tecla ser pressionada
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('o'):
        # Gera um nome de arquivo único usando um contador
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"captured_image_{timestamp}_{image_count}.jpg"
        file_path = os.path.join(save_directory, file_name)
        
        # Salva a imagem capturada
        cv2.imwrite(file_path, frame)
        print(f"Imagem salva como '{file_path}'")
        
        # Incrementa o contador
        image_count += 1
        
    elif key == ord('x'):
        print("Saindo.")
        break

# Libera a captura de vídeo e fecha todas as janelas
cam.release()
cv2.destroyAllWindows()
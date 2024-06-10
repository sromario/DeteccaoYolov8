import cv2
import os

# Carregar a imagem
image_path = 'img.jpeg'
img = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if img is None:
    print('Erro ao carregar a imagem.')
    exit()

# Diretório onde as cópias serão salvas
output_directory = 'dataset/data'

# Verificar se o diretório de saída existe; se não, criá-lo
os.makedirs(output_directory, exist_ok=True)

# Número de cópias desejadas
num_copies = 30

# Fazer cópias da imagem
for i in range(num_copies):
    # Nome do arquivo de saída
    output_filename = f'image_copy_{i}.jpeg'

    # Caminho completo para a cópia da imagem
    output_path = os.path.join(output_directory, output_filename)

    # Salvar a cópia da imagem
    cv2.imwrite(output_path, img)

    print(f'Cópia {i+1} salva em: {output_path}')

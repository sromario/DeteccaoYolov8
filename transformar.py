import os
import shutil
from sklearn.model_selection import train_test_split

# Diretórios
dataset_dir = 'dataset'
class_name = 'garrafa'  # Nome da sua classe

# Criar diretório da classe (se ainda não existir)
class_dir = os.path.join(dataset_dir, class_name)
os.makedirs(class_dir, exist_ok=True)

# Diretório onde estão as imagens
data_dir = os.path.join(dataset_dir, 'data')

# Listar todas as imagens
all_files = os.listdir(data_dir)

# Dividir os arquivos em treino e validação
train_files, validation_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Diretórios de treino e validação
train_dir = os.path.join(class_dir, 'train/img')
validation_dir = os.path.join(class_dir, 'validation/img')

# Criar diretórios de treino e validação, se não existirem
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Copiar arquivos de treino
for file_name in train_files:
    src = os.path.join(data_dir, file_name)
    dst = os.path.join(train_dir, file_name)
    shutil.copyfile(src, dst)

# Copiar arquivos de validação
for file_name in validation_files:
    src = os.path.join(data_dir, file_name)
    dst = os.path.join(validation_dir, file_name)
    shutil.copyfile(src, dst)

print("Processo concluído!")

import os
import shutil

source_root = '../lsun_bedroom_release'

target_root = '../organized_dataset'

real_dataset_name = '0_real'

if not os.path.exists(source_root):
    raise ValueError("Dataset root does not exist")

datasets = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]

for dataset in datasets:
    source_path = os.path.join(source_root, dataset)
    target_path = os.path.join(target_root, dataset)
    
    real_folder = os.path.join(target_path, '0_real')
    fake_folder = os.path.join(target_path, '1_fake')
    
    # 跳过0_real文件夹
    if dataset == real_dataset_name:
        continue  
    
    if not os.path.exists(real_folder):
        os.makedirs(real_folder)
    if not os.path.exists(fake_folder):
        os.makedirs(fake_folder)
    
    # 0_real文件夹
    real_source_path = os.path.join(source_root, real_dataset_name)
    for image_name in os.listdir(real_source_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(real_source_path, image_name)
            shutil.copy(image_path, real_folder)
    
    # 1_fake文件夹
    for image_name in os.listdir(source_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_path, image_name)
            shutil.copy(image_path, fake_folder)
print("ok")
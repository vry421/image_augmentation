import os
import shutil
from PIL import Image
from glob import glob
from tqdm import tqdm

data_dir = 'data'
temp_dir = 'temp'
temp_dup = 'temp_dup'

img_extensions = ('.jpg', '.jpeg', '.png')

def initial_rename(folder: str, image_list: list, repeats: int):

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    img_files = []

    working_folder = os.path.join(data_dir, folder)
    for extension in img_extensions:
        glob_files = glob(working_folder + '/*' + extension)
        img_files += glob_files

        print(f"Detected {len(glob_files)} files with {extension} extension")
    
    # Create temp directory for initial duplication

    if os.path.isdir(temp_dup):
        shutil.rmtree(temp_dup)
    os.mkdir(temp_dup)

    create_temp(img_files = img_files, temp_dir = temp_dup, number_name = False)

    img_files = glob(temp_dup + '/*' + '.png')

    if image_list:
        dup_files = [file for file in img_files if os.path.splitext(os.path.basename(file))[0] in image_list]
        print(f'From image duplication list, {len(dup_files)} files are detected in the dataset.')
        print(f'From image duplication list, {len(image_list) - len(dup_files)} files are not detected and will be ignored.')
        print(f'Duplication will be performed {repeats} times, leading to {len(dup_files) * repeats} duplicates.')

        for i in tqdm(range(repeats), desc='Duplicating Files', leave=True):
            for j, file in enumerate(dup_files):
                orig_name = os.path.splitext(os.path.basename(file))[0]
                new_name = orig_name + f'{(i + 1) * (j + 1)}'
                
                caption = caption = os.path.splitext(file)[0] + '.txt'

                img = Image.open(file)
                img.save(os.path.join(temp_dup, new_name + '.png'))

                shutil.copy(src = caption, dst = os.path.join(temp_dup, new_name + '.txt'))
    else:
        print("No initial duplication will be performed.")
    
    
    # Creation of Final Temporary Folder

    img_files = glob(temp_dup + '/*' + '.png')
    create_temp(img_files = img_files, temp_dir = temp_dir, number_name = True)

    # Cleanup
    shutil.rmtree(temp_dup)



def create_temp(img_files: list, temp_dir: str, number_name: bool):

    # Number name -> Converts filename to integer names. Suitable for LoRA training

    for i, file in enumerate(tqdm(img_files, desc='Creating Temporary Images', leave=True)):

        caption = os.path.splitext(file)[0] + '.txt'

        if number_name:
            new_name = str(i + 1)
        else:
            new_name = os.path.splitext(os.path.basename(file))[0]
        
        img = Image.open(file)
        img.save(os.path.join(temp_dir, new_name + '.png'))

        if os.path.exists(caption):
            shutil.copy(src = caption, dst = os.path.join(temp_dir, new_name + '.txt'))
        else:
            with open(os.path.join(temp_dir, new_name + '.txt'), 'w') as f:
                f.write("")


    
    


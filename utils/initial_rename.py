import os
import shutil
from PIL import Image
from glob import glob
from tqdm import tqdm

data_dir = 'data'
temp_dir = 'temp'

img_extensions = ('.jpg', '.jpeg', '.png')

def initial_rename(folder: str):

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    img_files = []

    working_folder = os.path.join(data_dir, folder)
    for extension in img_extensions:
        glob_files = glob(working_folder + '/*' + extension)
        img_files += glob_files

        print(f"Detected {len(glob_files)} files with {extension} extension")

    for i, file in enumerate(tqdm(img_files, desc='Creating Temporary Images', leave=True)):

        new_name = str(i + 1)
        caption = os.path.splitext(file)[0] + '.txt'

        file_name = os.path.basename(file)
        caption_name = os.path.basename(caption)
        
        img = Image.open(file)
        img.save(os.path.join(temp_dir, new_name + '.png'))

        if os.path.exists(caption):
            shutil.copy(src = caption, dst = os.path.join(temp_dir, new_name + '.txt'))
        else:
            with open(os.path.join(temp_dir, new_name + '.txt'), 'w') as f:
                f.write("")
            

    
    


import os
import cv2
import numpy as np
import imutils
import shutil
import random
import csv
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from utils import initial_rename

# Note to self: img.shape -> (Height, Width, Channel)

temp_dir = 'temp'
output_dir = 'output'

DEFAULT_CONFIG = {

    # Parameters to be changed -----------------------------------------

    'INPUT_FOLDER': 'sample', # Folder should be inside /data
    'OUTPUT_FOLDER': 'out', # Folder will be placed inside /output
    'NUM_REPEATS': 10,
    'NUM_ACTIVATION_TAGS': 1,

    'REPLACE_CHAR_FROM': '_', # Replace FROM to TO. Ex: "green_eyes -> green eyes". Replace both to None if not needed
    'REPLACE_CHAR_TO': ' ',  # DOES NOT REPLACE CHARACTER IN ACTIVATION TAGS!

    # ------------------------------------------------------------------

    'SHUFFLE_CAPTIONS': 1,

    'PROB_DUPLICATE': 0.1, # Probability of only duplicating an image
    'PROB_HFLIP': 0.5,
    'PROB_VFLIP': 0.3,
    'PROB_GRAYSCALE': 0.5,
    'PROB_ROTATE': 0.5,
    'PROB_ROTATE_BOUNDS': 0.5, # Rotate bounds ensures that entirety of image is visible after rotation, but introduces larger black borders
    'PROB_BRIGHTNESS_CONTRAST': 0.7,
    'PROB_TRANSLATE': 0.3,
    'PROB_JITTER': 0.7,
    'PROB_SKETCH': 0.2,
    'PROB_LINEART': 0.3,
    'PROB_CROP': 0.2,
    'PROB_NOISE': 0.7,

    'RANGE_ROTATE': (-15, 15),
    'RANGE_BRIGHTNESS': (5, 20),
    'RANGE_CONTRAST': (0.8, 1.2),
    'RANGE_TRANSLATE_X': (-0.3, 0.3), # % of Width
    'RANGE_TRANSLATE_Y': (-0.2, 0.2), # % of Height
    'RANGE_HUE': (-20, 20),
    'RANGE_SATURATION': (-10, 10),
    'RANGE_SOBEL_KERNEL': (3, 5), # FOR LINEART | Should be odd
    'RANGE_CROP_X': (0.9, 1), # % of Width
    'RANGE_CROP_Y': (0.9, 1), # % of Height
    'RANGE_SKETCH_SIGMAX': (0, 7),
    'RANGE_SKETCH_KERNEL': (3, 7),
    'RANGE_SP_FRACTION': (0.01, 0.07), # % of total pixels on which salt and pepper noise is applied
    'RANGE_GAUSSIAN_MEAN': (0, 5),
    'RANGE_GAUSSIAN_STD': (1, 5),
    'RANGE_GAUSSIAN_ALPHA': (0.7, 0.9),

    # For tags, replace with "" as needed
    'TAG_VFLIP': ', upside-down',
    'TAG_GRAYSCALE': ', grayscale',
    'TAG_ROTATE': ', black border',
    'TAG_TRANSLATE': ', black bars',
    'TAG_LINEART': ', lineart',
    'TAG_SKETCH': ', sketch',
    
}

class ImageAugmentation():

    def __init__(self, CONFIG: dict = DEFAULT_CONFIG):

        self.INPUT_FOLDER = CONFIG['INPUT_FOLDER']
        self.OUTPUT_FOLDER = CONFIG['OUTPUT_FOLDER']
        self.NUM_REPEATS = CONFIG['NUM_REPEATS']
        self.NUM_ACTIVATION_TAGS = CONFIG['NUM_ACTIVATION_TAGS']
        self.REPLACE_CHAR_FROM = CONFIG['REPLACE_CHAR_FROM']
        self.REPLACE_CHAR_TO = CONFIG['REPLACE_CHAR_TO']

        self.SHUFFLE_CAPTIONS = CONFIG['SHUFFLE_CAPTIONS']
        self.PROB_DUPLICATE = CONFIG['PROB_DUPLICATE']
        self.PROB_HFLIP = CONFIG['PROB_HFLIP']
        self.PROB_VFLIP = CONFIG['PROB_VFLIP']
        self.PROB_GRAYSCALE = CONFIG['PROB_GRAYSCALE']
        self.PROB_ROTATE = CONFIG['PROB_ROTATE']
        self.PROB_ROTATE_BOUNDS = CONFIG['PROB_ROTATE_BOUNDS']
        self.PROB_BRIGHTNESS_CONTRAST = CONFIG['PROB_BRIGHTNESS_CONTRAST']
        self.PROB_TRANSLATE = CONFIG['PROB_TRANSLATE']
        self.PROB_JITTER = CONFIG['PROB_JITTER']
        self.PROB_SKETCH = CONFIG['PROB_SKETCH']
        self.PROB_LINEART = CONFIG['PROB_LINEART']
        self.PROB_CROP = CONFIG['PROB_CROP']
        self.PROB_NOISE = CONFIG['PROB_NOISE']
        
        self.TAG_VFLIP = CONFIG['TAG_VFLIP']
        self.TAG_GRAYSCALE = CONFIG['TAG_GRAYSCALE']
        self.TAG_ROTATE = CONFIG['TAG_ROTATE']
        self.TAG_TRANSLATE = CONFIG['TAG_TRANSLATE']
        self.TAG_LINEART = CONFIG['TAG_LINEART']
        self.TAG_SKETCH = CONFIG['TAG_SKETCH']

        self.RANGE_ROTATE = CONFIG['RANGE_ROTATE']
        self.RANGE_BRIGHTNESS = CONFIG['RANGE_BRIGHTNESS']
        self.RANGE_CONTRAST = CONFIG['RANGE_CONTRAST']
        self.RANGE_TRANSLATE_X = CONFIG['RANGE_TRANSLATE_X']
        self.RANGE_TRANSLATE_Y = CONFIG['RANGE_TRANSLATE_Y']
        self.RANGE_HUE = CONFIG['RANGE_HUE']
        self.RANGE_SATURATION = CONFIG['RANGE_SATURATION']
        self.RANGE_SOBEL_KERNEL = CONFIG['RANGE_SOBEL_KERNEL']
        self.RANGE_CROP_X = CONFIG['RANGE_CROP_X']
        self.RANGE_CROP_Y = CONFIG['RANGE_CROP_Y']
        self.RANGE_SKETCH_SIGMAX = CONFIG['RANGE_SKETCH_SIGMAX']
        self.RANGE_SKETCH_KERNEL = CONFIG['RANGE_SKETCH_KERNEL']
        self.RANGE_SP_FRACTION = CONFIG['RANGE_SP_FRACTION']
        self.RANGE_GAUSSIAN_MEAN = CONFIG['RANGE_GAUSSIAN_MEAN']
        self.RANGE_GAUSSIAN_STD = CONFIG['RANGE_GAUSSIAN_STD']
        self.RANGE_GAUSSIAN_ALPHA = CONFIG['RANGE_GAUSSIAN_ALPHA']

        initial_rename(self.INPUT_FOLDER)
        

    def __hflip(self, img, logs):

        logs['count']['hflip'] += 1
        img = cv2.flip(src = img, flipCode = 1)

        return img, logs


    def __vflip(self, img, logs, caption):

        logs['count']['vflip'] += 1
        img = cv2.flip(src = img, flipCode = 0)

        with open(caption, 'a') as f:
            f.write(self.TAG_VFLIP)

        return img, logs


    def __grayscale(self, img, logs, caption):

        logs['count']['grayscale'] += 1
        img = cv2.cvtColor(src = img, code = cv2.COLOR_RGB2GRAY)

        with open(caption, 'a') as f:
            f.write(self.TAG_GRAYSCALE)

        return img, logs
    

    def __rotate(self, img, logs, caption):

        angle = random.randrange(start = self.RANGE_ROTATE[0], stop = self.RANGE_ROTATE[1])

        logs['count']['rotate'] += 1
        logs['hist']['rotate'].append(angle)

        if random.random() < self.PROB_ROTATE_BOUNDS:
            img = imutils.rotate_bound(image = img, angle = angle)
        else:
            img = imutils.rotate(image = img, angle = angle)

        with open(caption, 'a') as f:
            f.write(self.TAG_ROTATE)

        return img, logs
    

    def __brightness_contrast(self, img, logs):

        brightness = random.randrange(start = self.RANGE_BRIGHTNESS[0], stop = self.RANGE_BRIGHTNESS[1])
        contrast = round(random.uniform(a = self.RANGE_CONTRAST[0], b = self.RANGE_CONTRAST[1]), 2)

        logs['count']['brightness+contrast'] += 1
        logs['hist']['brightness'].append(brightness)
        logs['hist']['contrast'].append(contrast)

        img = cv2.convertScaleAbs(src = img, alpha = contrast, beta = brightness)

        return img, logs
        

    def __translate(self, img, logs, caption):

        x_pct = round(random.uniform(a = self.RANGE_TRANSLATE_X[0], b = self.RANGE_TRANSLATE_X[1]), 2)
        y_pct = round(random.uniform(a = self.RANGE_TRANSLATE_Y[0], b = self.RANGE_TRANSLATE_Y[1]), 2)

        x_shift = int(x_pct * img.shape[1])
        y_shift = int(y_pct * img.shape[0])

        logs['count']['translate'] += 1
        logs['hist']['translate_x'].append(x_pct)
        logs['hist']['translate_y'].append(y_pct)

        T = np.float32([
            [1, 0, x_shift],
            [0, 1, y_shift]
        ])

        shifted = cv2.warpAffine(src = img, M = T, dsize = (img.shape[1], img.shape[0]))

        with open(caption, 'a') as f:
            f.write(self.TAG_TRANSLATE)

        return shifted, logs   


    def __jitter(self, img, logs):
        
        h_add = random.randrange(start = self.RANGE_HUE[0], stop = self.RANGE_HUE[1])
        s_add = random.randrange(start = self.RANGE_SATURATION[0], stop = self.RANGE_SATURATION[1])

        logs['count']['jitter'] += 1
        logs['hist']['hue'].append(h_add)
        logs['hist']['saturation'].append(s_add)

        img = cv2.cvtColor(src = img, code = cv2.COLOR_RGB2HSV)

        h, s, v = np.float32(cv2.split(img))

        h = np.clip(a = h + h_add, a_min = 0, a_max = 179)
        s = np.clip(a = s + s_add, a_min = 0, a_max = 255)

        img = cv2.merge(np.uint8([h, s, v]))
        img = cv2.cvtColor(src = img, code = cv2.COLOR_HSV2RGB)

        return img, logs
    

    def __lineart(self, img, logs, caption):

        # Lineart uses Sobel edge detection

        dx = random.randint(a = 0, b = 1)
        dy = random.randint(a = 0, b = 1)

        kernel = random.randrange(start = self.RANGE_SOBEL_KERNEL[0], stop = self.RANGE_SOBEL_KERNEL[1], step = 2)

        if dx + dy == 0:
            dx = 1
            dy = 1

        logs['count']['lineart'] += 1
        logs['hist']['sobel_dx'].append(dx)
        logs['hist']['sobel_dy'].append(dy)
        logs['hist']['sobel_kernel'].append(dy)

        img = cv2.Sobel(src = img, ddepth = cv2.CV_8U, dx = dx, dy = dy, ksize = kernel)


        with open(caption, 'a') as f:
            f.write(self.TAG_LINEART)

        return img, logs


    def __sketch(self, img, logs, caption):

        ksize = random.randrange(start = self.RANGE_SKETCH_KERNEL[0], stop = self.RANGE_SKETCH_KERNEL[1], step = 2)
        sigmaX = random.randint(a = self.RANGE_SKETCH_SIGMAX[0], b = self.RANGE_SKETCH_SIGMAX[1])

        logs['count']['sketch'] += 1
        logs['hist']['sketch_kernel'].append(ksize)
        logs['hist']['sketch_sigmax'].append(sigmaX)

        grey = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)
        temp = cv2.bitwise_not(src = grey)
        temp = cv2.GaussianBlur(src = temp, ksize = (ksize, ksize), sigmaX = sigmaX)
        temp = cv2.bitwise_not(temp)

        img = cv2.divide(src1 = grey, src2 = temp, scale=256)
        

        with open(caption, 'a') as f:
            f.write(self.TAG_SKETCH)

        return img, logs
    

    def __crop(self, img, logs):


        x_pct = round(random.uniform(a = self.RANGE_CROP_X[0], b = self.RANGE_CROP_X[1]), 2)
        y_pct = round(random.uniform(a = self.RANGE_CROP_Y[0], b = self.RANGE_CROP_Y[1]), 2)

        x_box = int(x_pct * img.shape[1])
        y_box = int(y_pct * img.shape[0])

        logs['count']['crop'] += 1
        logs['hist']['crop_x'].append(x_pct)
        logs['hist']['crop_y'].append(y_pct)

        x1 = np.random.randint(low = 0, high = img.shape[1] - x_box + 1)
        y1 = np.random.randint(low = 0, high = img.shape[0] - y_box + 1)

        img = img[x1 : x1 + x_box, y1 : y1 + y_box]

        return img, logs
    

    def __add_noise(self, img, logs):

        if random.random() < 0.5: # Salt and Pepper
            
            noise_fraction = random.uniform(a = self.RANGE_SP_FRACTION[0], b = self.RANGE_SP_FRACTION[1])

            logs['count']['noise_saltpepper'] += 1
            logs['hist']['noise_saltpepper_fraction'].append(noise_fraction)

            height, width, _ = img.shape

            max_pixels = height * width
            noise_pixels = int(max_pixels * noise_fraction)

            for i in range(noise_pixels):
                y = random.randint(0, height - 1)
                x = random.randint(0, width - 1)

                img[y][x] = random.choice([0, 255])

        else: # Gaussian Blur

            mean = random.uniform(a = self.RANGE_GAUSSIAN_MEAN[0], b = self.RANGE_GAUSSIAN_MEAN[1])
            std = random.uniform(a = self.RANGE_GAUSSIAN_STD[0], b = self.RANGE_GAUSSIAN_STD[1])
            alpha = random.uniform(a = self.RANGE_GAUSSIAN_ALPHA[0], b = self.RANGE_GAUSSIAN_ALPHA[1])

            logs['count']['noise_gaussian'] += 1
            logs['hist']['noise_gaussian_mean'].append(mean)
            logs['hist']['noise_gaussian_std'].append(std)
            logs['hist']['noise_gaussian_alpha'].append(alpha)

            noise = np.random.normal(loc = mean, scale = std, size = img.shape).astype(np.uint8)

            img = cv2.addWeighted(src1 = img, alpha = alpha, src2 = noise, beta = 1 - alpha, gamma = 0)


        return img, logs
    

    def __move_to_output(self):

        img_files = sorted(glob(temp_dir + '/*.png'))
        caption_files = sorted(glob(temp_dir + '/*.txt'))

        all_files = img_files + caption_files

        final_output_folder = os.path.join(output_dir, self.OUTPUT_FOLDER)
        if os.path.isdir(final_output_folder):
            shutil.rmtree(final_output_folder)
        os.mkdir(final_output_folder)


        for file in tqdm(all_files, desc='Moving files to output directory', leave=True):
            shutil.move(src = file, dst = final_output_folder)

        shutil.rmtree(temp_dir)


    def __replace_char(self, file):

        with open(file, 'r') as f:

            caption_list = list(csv.reader(f))
            
            if len(caption_list) != 0:
            
                activation = caption_list[0][:self.NUM_ACTIVATION_TAGS]
                captions = caption_list[0][self.NUM_ACTIVATION_TAGS:]
                captions = [word.replace(self.REPLACE_CHAR_FROM, self.REPLACE_CHAR_TO) for word in captions]
                caption_list = activation + captions              
                
        os.remove(file)

        with open(file, 'w') as f:

            txtfile = csv.writer(f)
            txtfile.writerow(caption_list)       


    def augment(self) -> dict:

        logs = {}

        logs['count'] = {
            'hflip': 0,
            'vflip': 0,
            'grayscale': 0,
            'rotate': 0,
            'brightness+contrast': 0,
            'translate': 0,
            'jitter': 0,
            'sketch': 0,
            'lineart': 0,
            'crop': 0,
            'duplicate': 0,
            'noise_saltpepper': 0,
            'noise_gaussian': 0,
        }

        logs['hist'] = {
            'rotate': [],
            'brightness': [],
            'contrast': [],
            'translate_x': [],
            'translate_y': [],
            'hue': [],
            'saturation': [],
            'sobel_dx': [],
            'sobel_dy': [],
            'sobel_kernel': [],
            'crop_x': [],
            'crop_y': [],
            'sketch_kernel': [],
            'sketch_sigmax': [],
            'noise_saltpepper_fraction': [],
            'noise_gaussian_mean': [],
            'noise_gaussian_std': [],
            'noise_gaussian_alpha': [],
        }


        img_files = sorted(glob(temp_dir + '/*.png'))
        caption_files = sorted(glob(temp_dir + '/*.txt'))
        num_img = len(img_files)
        num_aug = num_img * self.NUM_REPEATS

        print(f"Number of images in dataset: {num_img}")
        print(f"Number of augmented images: {num_aug}")
        print(f"Total number of output images: {num_img + num_aug}")

        for i, file in enumerate(tqdm(img_files, desc=f'Ensuring correct RGB Profile', leave=True)):
            
            cwd = os.getcwd()
            filepath = os.path.join(cwd, file)

            subprocess.call(args = ['convert', filepath, filepath])

        if self.REPLACE_CHAR_FROM is not None and self.REPLACE_CHAR_TO is not None:
            print(f"Character Replacement Activated. Replacing [{self.REPLACE_CHAR_FROM}] to [{self.REPLACE_CHAR_TO}]")
            for file in tqdm(caption_files, desc='Replacing Characters in Captions', leave=True):
                self.__replace_char(file)

        for i in tqdm(range(self.NUM_REPEATS), 'Batch Progress', leave=True):
            for j, file in enumerate(tqdm(img_files, desc=f'Performing Augmentation on Batch {i + 1}', leave=False)):
                
                BYPASS_FROM_SKETCH = 0
                BYPASS_FROM_LINEART = 0

                new_name = str((i + 1) * num_img + (j + 1))
                new_img = os.path.join(temp_dir, new_name + '.png')
                new_caption = os.path.join(temp_dir, new_name + '.txt')

                shutil.copy(src = img_files[j], dst = new_img)
                shutil.copy(src = caption_files[j], dst = new_caption)

                img = cv2.imread(new_img)
                

                ## DUPLICATE
                if random.random() < self.PROB_DUPLICATE:
                    logs['count']['duplicate'] += 1
                    continue
 
                ## COLOR JITTER

                if random.random() < self.PROB_JITTER:
                    img, logs = self.__jitter(img, logs)


                ## SKETCH

                if random.random() < self.PROB_SKETCH:
                    img, logs = self.__sketch(img, logs, new_caption)
                    BYPASS_FROM_SKETCH = 1


                ## SOBEL LINEART

                if random.random() < self.PROB_LINEART and not BYPASS_FROM_SKETCH:
                    img, logs = self.__lineart(img, logs, new_caption)
                    BYPASS_FROM_LINEART = 1

                
                ## NOISE

                if random.random() < self.PROB_NOISE and not (BYPASS_FROM_SKETCH or BYPASS_FROM_LINEART):
                    img, logs = self.__add_noise(img, logs)  


                ## FLIP

                if random.random() < self.PROB_HFLIP:
                    img, logs = self.__hflip(img, logs)
                
                if random.random() < self.PROB_VFLIP:
                    img, logs = self.__vflip(img, logs, new_caption)


                ## BRIGHTNESS AND CONTRAST

                if random.random() < self.PROB_BRIGHTNESS_CONTRAST:
                    img, logs = self.__brightness_contrast(img, logs)               


                ## TRANSLATION

                if random.random() < self.PROB_TRANSLATE:
                    img, logs = self.__translate(img, logs, new_caption)


                ## ROTATION

                if random.random() < self.PROB_ROTATE:
                    img, logs = self.__rotate(img, logs, new_caption)
                
 
                ## RANDOM CROP

                if random.random() < self.PROB_CROP:
                    img, logs = self.__crop(img, logs)


                # GRAYSCALE

                if random.random() < self.PROB_GRAYSCALE and not BYPASS_FROM_SKETCH:
                    img, logs = self.__grayscale(img, logs, new_caption)


                ## Write Final Image
                cv2.imwrite(filename = new_img, img=img)


                ## Shuffle Captions

                if self.SHUFFLE_CAPTIONS:
                    
                    with open(new_caption, 'r') as f:

                        caption_list = list(csv.reader(f))
                        
                        if len(caption_list) == 0:
                            continue
                        
                        activation = caption_list[0][:self.NUM_ACTIVATION_TAGS]
                        captions = caption_list[0][self.NUM_ACTIVATION_TAGS:]

                        random.shuffle(captions)

                        caption_list = activation + captions
                    
                    os.remove(new_caption)


                    with open(new_caption, 'w') as f:

                        txtfile = csv.writer(f)
                        txtfile.writerow(caption_list)
                

        ## Move files to output folder
        self.__move_to_output()


        return logs

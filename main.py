import warnings
warnings.filterwarnings('ignore')

from utils import ImageAugmentation, plot_counts

# Before usage, install ImageMagick

CONFIG = {

    # Parameters to be changed -----------------------------------------

    'INPUT_FOLDER': 'sample', # Folder should be inside /data
    'OUTPUT_FOLDER': 'out', # Folder will be placed inside /output
    'NUM_REPEATS': 5,
    'NUM_ACTIVATION_TAGS': 1,

    'REPLACE_CHAR_FROM': '_', # Replace FROM to TO. Ex: "green_eyes -> green eyes". Replace both to None if not needed
    'REPLACE_CHAR_TO': ' ',  # DOES NOT REPLACE CHARACTER IN ACTIVATION TAGS!

    'IMAGE_SHUFFLE': 1, # Shuffle images before augmentation
    'IMAGE_DUPLICATE_LIST': [
        'cat', 
        'rat',
    ], # List of image files to duplicate prior to augmentation. Only pass filenames WITHOUT EXTENSION!
       # Duplication is done PRIOR TO AUGMENTATION
    'IMAGE_DUPLICATE_REPEAT': 1, # Number of times the images in the list is duplicated

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

def main():

    img_aug = ImageAugmentation(CONFIG)
    logs = img_aug.augment()

    plot_counts(logs['count'])

if __name__ == '__main__':
    main()
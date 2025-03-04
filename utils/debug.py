import csv
from glob import glob
from tqdm import tqdm

def get_caption(folder: str, show_captions: bool = False) -> dict:

    """
    Returns a dictionary containing captions per file in a folder
    """

    num_captions = len(glob(folder + '/*.txt'))

    caption_dict = {}

    print(num_captions)

    for file_num in tqdm(range(1, num_captions + 1), desc='Getting Captions', leave=True):
    
        filename = folder + '/' + str(file_num) + '.txt'

        with open(filename, 'r') as file:

            caption = list(csv.reader(file))[0]

            caption_dict[file_num] = caption
        
    if show_captions:
        for key, value in caption_dict.items():
            print(f"{key}: {value} \n")

    return caption_dict

import os.path
import sys

if os.path.exists('input.txt'):
    sys.stdin = open( 'input.txt','r')
    sys.stdout = open('output.txt','w')
from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr30k',
                       karpathy_json_path='C:/Users/susha/ImageCaption-Transformer/caption data/dataset_flickr30k.json',
                       image_folder='C:/Users/susha/ImageCaption-Transformer/caption data/flickr30k_images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='C:/Users/susha/ImageCaption-Transformer/caption data/',
                       max_len=196)

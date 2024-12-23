import torch
from torch.utils.data import Dataset
import h5py
import json
import os
data_folder = '/content/caption_data/'  # folder with data files saved by create_input_files.py
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

class CaptionDataset(Dataset):

    def __init__(self, data_folder, data_name, split, transform=None):

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        
        with open(os.path.join(data_folder,  'WORDMAP_' + data_name + '.json'), 'r') as j:
            self.word_map = json.load(j)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        self.pad_token = torch.tensor([self.word_map['<pad>']], dtype=torch.int64)
   

        # print(self.pad_token)
        # print(self.pad_token.shape)


        # Load labels (completely into memory)
        with open(os.path.join(data_folder, self.split + '_LABELS_' + data_name + '.json'), 'r') as j:
            self.labels = json.load(j)
        

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        # print((caption != self.pad_token).unsqueeze(0).int())   #(1, 196)
        decoder_mask=(caption != self.pad_token).unsqueeze(0).int() & causal_mask(caption.size(0)) # (1, 196, 196) 
        # print(causal_mask(caption.size(0)))
        # print(decoder_mask.shape)

        label = torch.LongTensor([self.labels[i]])
        caplen = torch.LongTensor([self.caplens[i]])


        if self.split is 'TRAIN':
            return img, caption, label,decoder_mask,caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, label, all_captions,decoder_mask,caplen

    def __len__(self):
        return self.dataset_size

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
# c=CaptionDataset(data_folder, data_name, 'TRAIN')
# hi=CaptionDataset(data_folder, data_name, 'TRAIN')
# hi.__getitem__(0)


import os
import os.path as osp
import glob
import pickle
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF

from torch.utils import data
from torchvision import transforms as T
from PIL import Image



def load_pickle(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


class DataSet(data.Dataset):

    def __init__(self, config, transform, mode):


        assert mode in ['train', 'test']
        self.transform = transform
        self.mode = mode
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR']) # , config['TRAINING_CONFIG']['MODE']
        self.H, self.W = config['MODEL_CONFIG']['IMG_SIZE'].split(",")
        self.H, self.W = int(self.H), int(self.W)
        self.img_size = (self.H, self.W, 3)
        self.domain = config['TRAINING_CONFIG']['DOMAIN']
        print(f'mode : {self.mode}, domain : {self.domain}')
        #self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))

        if self.domain == 'category':
            plk_path = osp.join('label', f'category_segment_{self.mode}.plk')
            self.data_list = load_pickle(plk_path)
        elif self.domain == 'color':
            plk_path = osp.join('label', f'color_segment_{self.mode}.plk')
            self.data_list = load_pickle(plk_path)

        print(f'num of data : {len(self.data_list)}')


    def transform_func(self, image, mask):
        # Resize
        resize = T.Resize((self.H, self.W))
        image = resize(image)
        mask = resize(mask.unsqueeze(0)).squeeze()

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return self.transform(image), mask

    def __getitem__(self, index):
        data = self.data_list[index]
        image_name = data['img_name']
        mask = data['semseg']

        image = Image.open(osp.join(self.img_dir, f'{image_name}')).convert('RGB')
        mask = torch.from_numpy(mask.astype(np.uint8)).long()

        image, mask = self.transform_func(image, mask)
        return image, mask

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config, mode):

    img_transform = list()
    #H, W = config['MODEL_CONFIG']['IMG_SIZE'].split(",")
    #img_transform.append(T.Resize((H, W)))
    #img_transform.append(T.RandomHorizontalFlip(p=0.5))
    #img_transform.append(T.RandomVerticalFlip(p=0.5))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    if mode == 'train':
        batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']
    else:
        batch_size = 1

    dataset = DataSet(config, img_transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader

import os
import time
import datetime
import torch
import torch.nn as nn
import glob
import os.path as osp
import numpy as np
import cv2

from model import Unet
from torchvision.utils import save_image
from data_loader import get_loader

# decode segment
# https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/


class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""
        self.train_data_loader = get_loader(config, 'train')
        self.test_data_loader = get_loader(config, 'test')

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.lr          = float(config['TRAINING_CONFIG']['LR'])
        self.lambda_cls = config['TRAINING_CONFIG']['LAMBDA_CLS']
        self.domain = config['TRAINING_CONFIG']['DOMAIN']

        self.color_cate = ['bk', 'beige', 'black', 'blue', 'brown',
                           'gray', 'green', 'orange', 'pink', 'purple',
                           'red', 'white', 'yellow']
        # bgr
        self.color_rgb = np.array(
            [(255, 255, 255),# 0=background
             (198, 226, 244),# 1=beige
             (0, 0, 0), # 2=black
             (255, 32, 0), # 3=blue
             (77, 96, 128), #4=brown
             (128, 128, 128), # gray
             (110, 198, 119), # green
             (0, 131, 255), # orange
             (193, 182, 255), # pink
             (169, 81, 120), # purple
             (36, 28, 237), # red
             (239, 243, 243), # white
             (0, 255, 255), # yellow
             ])

        self.cloth_cate = ['bk', 'T-shirt', 'bag', 'belt', 'blazer',
                           'blouse', 'coat', 'dress', 'face', 'hair',
                           'hat', 'jeans', 'legging', 'pants', 'scarf',
                           'shoe', 'shorts', 'skin', 'skirt', 'socks',
                           'stocking', 'sunglass', 'sweater']
        self.cloth_rgb = np.array(
            [(255, 255, 255),# 0=background
             (198, 226, 244),# 1=t-shirt
             (0, 0, 0), # 2=bag
             (255, 32, 0), # 3=belt
             (77, 96, 128), #4=blazer
             (128, 128, 128), # blouse
             (110, 198, 119), # coat
             (0, 131, 255), # dress
             (193, 182, 255), # face
             (169, 81, 120), # hair
             (36, 28, 237), # hat
             (239, 243, 243), # jeans
             (0, 255, 255), # legging
             (128, 128, 0), # pants
             (255, 255, 0), # scarf
             (102, 153, 51), # shoe
             (0, 0, 128), # shorts
             (153, 204, 255), # skin
             (255, 255, 204), # skirt
             (102, 0, 102), # socks
             (255, 204, 204), # stocking
             (51, 51, 51), # sunglass
             (153, 102, 102), # sunglass
             ])

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.loss = nn.CrossEntropyLoss()

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.test_step    = config['TRAINING_CONFIG']['TEST_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):

        if self.domain == 'category':
            self.model = Unet(n_channels=3, n_classes=23).to(self.gpu)
        elif self.domain == 'color':
            self.model = Unet(n_channels=3, n_classes=13).to(self.gpu)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, (self.beta1, self.beta2))
        self.print_network(self.model, 'Model')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*-model.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x.split('-')[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        model_path = os.path.join(self.model_dir, '{}-model.ckpt'.format(epoch))
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(self.gpu)
        return epoch

    def decode_segmap(self, image, nc=21):

        if self.domain == 'category':
            label_colors = self.cloth_rgb
        else:
            label_colors = self.color_rgb

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=0)
        rgb = np.squeeze(rgb)
        return np.transpose(rgb, (1, 2, 0))

    def train(self):

        # Set data loader.
        train_data_loader = self.train_data_loader
        iterations = len(self.train_data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(train_data_loader)
        fixed_image, fixed_mask = next(data_iter)

        fixed_image = fixed_image.to(self.gpu)
        fixed_mask = fixed_mask.to(self.gpu)

        start_epoch = self.restore_model()
        start_time = time.time()
        print('Start training...')
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    images, target = next(data_iter)
                except:
                    data_iter = iter(train_data_loader)
                    images, target = next(data_iter)

                images = images.to(self.gpu)
                target = target.to(self.gpu)

                loss_dict = dict()

                pred_edge = self.model(images)
                loss = self.loss(pred_edge, target)

                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict['loss'] = loss.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():

                    z_f = len(str(self.epoch))
                    epoch_str = str(e + 1).zfill(z_f)
                    sample_path = osp.join(self.sample_dir, epoch_str)
                    os.makedirs(sample_path, exist_ok=True)

                    for i, data in enumerate(zip(fixed_image.chunk(self.batch_size, dim=0), fixed_mask.chunk(self.batch_size, dim=0))):
                        image, mask = data
                        image, mask = image.to(self.gpu), mask.to(self.gpu)
                        out = self.model(image)
                        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
                        img_np = self.denorm(image).squeeze().detach().cpu().numpy()
                        img_np = img_np.astype(np.uint8) * 255
                        img_np = np.transpose(img_np, (1, 2, 0))
                        img_np[:, :, [0, 1, 2]] = img_np[:, :, [2, 1, 0]]
                        pred_rbg = self.decode_segmap(om)
                        mask_rbg = self.decode_segmap(mask.detach().cpu().numpy())
                        concat_img = cv2.hconcat([img_np, mask_rbg, pred_rbg])
                        cv2.imwrite(osp.join(sample_path, f'{i}_fixed_result.jpg'), concat_img)

                    print('Saved real and fake images into {}...'.format(self.sample_dir))

            if (e + 1) % self.test_step == 0:
                self.test(self.test_data_loader, e + 1, 'test')

            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                model_path = os.path.join(self.model_dir, '{}-model.ckpt'.format(e + 1))
                torch.save(self.model.state_dict(), model_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def test(self, data_loader, epoch, mode='test'):

        z_f = len(str(self.epoch))
        epoch_str = str(epoch).zfill(z_f)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                image, mask = data
                image, mask = image.to(self.gpu), mask.to(self.gpu)
                out = self.model(image)
                om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
                img_np = self.denorm(image).squeeze().detach().cpu().numpy()
                img_np = img_np.astype(np.uint8) * 255
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np[:, :, [0, 1, 2]] = img_np[:, :, [2, 1, 0]]
                pred_rbg = self.decode_segmap(om)
                mask_rbg = self.decode_segmap(mask.detach().cpu().numpy())
                concat_img = cv2.hconcat([img_np, mask_rbg, pred_rbg])
                cv2.imwrite(osp.join(self.result_dir, f'{mode}_{epoch_str}_{i}_result.jpg'), concat_img)






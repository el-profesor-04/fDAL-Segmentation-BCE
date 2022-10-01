import cv2
import torch
from time import time

from efficientnet_pytorch import EfficientNet
#from matplotlib import pyplot as plt
#from tensorboardX import SummaryWriter
import numpy as np
import os
import json
import math
from transforms3d.euler import euler2mat
from PIL import Image
from model import compile_model
from tools import SimpleLoss, get_batch_iou, normalize_img, img_transform, get_val_info


from tqdm import tqdm
from core import fDALLearner   # core.py
from torchvision import transforms
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader
#from data_list import ImageList, ForeverDataIterator
import torch.optim as optim
import random
import fire
import torch.nn.utils.spectral_norm as sn


from data import compile_data
import itertools



def get_camera_info(translation, rotation, sensor_options):
    roll = math.radians(rotation[2] - 90)
    pitch = -math.radians(rotation[1])
    yaw = -math.radians(rotation[0])
    rotation_matrix = euler2mat(roll, pitch, yaw)

    calibration = np.identity(3)
    calibration[0, 2] = sensor_options['image_size_x'] / 2.0
    calibration[1, 2] = sensor_options['image_size_y'] / 2.0
    calibration[0, 0] = calibration[1, 1] = sensor_options['image_size_x'] / (
            2.0 * np.tan(sensor_options['fov'] * np.pi / 360.0))

    return torch.tensor(rotation_matrix), torch.tensor(translation), torch.tensor(calibration)


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, record_path, data_aug_conf, ticks):
        self.record_path = record_path
        self.data_aug_conf = data_aug_conf
        self.ticks = ticks

        with open(os.path.join(self.record_path, 'sensors.json'), 'r') as f:
            self.sensors_info = json.load(f)

    def __len__(self):
        return self.ticks

    def __getitem__(self, idx):
        imgs = []
        img_segs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        binimgs = Image.open(os.path.join(self.record_path + "birds_view_semantic_camera", str(idx) + '.png'))
        binimgs = binimgs.crop((25, 25, 175, 175))
        binimgs = binimgs.resize((200, 200))
        binimgs = np.array(binimgs)
        binimgs = torch.tensor(binimgs).permute(2, 1, 0)[0]
        binimgs = binimgs[None, :, :]/255

        for sensor_name, sensor_info in self.sensors_info['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(self.record_path + sensor_name, str(idx) + '.png'))
                image_seg = Image.open(os.path.join(self.record_path + sensor_name + "_semantic", str(idx) + '.png'))

                tran = sensor_info["transform"]["location"]
                rot = sensor_info["transform"]["rotation"]
                sensor_options = sensor_info["sensor_options"]

                rot, tran, intrin = get_camera_info(tran, rot, sensor_options)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                img_seg, _, _ = img_transform(image_seg, post_rot, post_tran,
                                              resize=resize,
                                              resize_dims=resize_dims,
                                              crop=crop,
                                              flip=flip,
                                              rotate=rotate, )

                img, post_rot2, post_tran2 = img_transform(image, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate, )

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                img_seg = np.array(img_seg)
                img_seg = torch.tensor(img_seg).permute(2, 0, 1)[0]
                img_seg = img_seg[None, :, :]

                imgs.append(normalize_img(img))
                # img_segs.append(normalize_img(img_seg))
                img_segs.append(img_seg/255)
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)
        #print(len(imgs),'in carla dataset',torch.stack(imgs).shape,'stack???')
        #print('carla dims')
        #print(torch.stack(rots).shape, torch.stack(trans).shape,
        #        torch.stack(intrins).shape, torch.stack(post_rots).shape, torch.stack(post_trans).shape)
        #print(torch.stack(imgs).shape)
        #print('carla ......')

        return (torch.stack(imgs).float(), torch.stack(img_segs).float(), torch.stack(rots).float(), torch.stack(trans).float(),
                torch.stack(intrins).float(), torch.stack(post_rots).float(), torch.stack(post_trans).float(), binimgs.float())

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0

        return resize, resize_dims, crop, flip, rotate


'''def get_val(model, val_loader, device, loss_fn, type):
    model.eval()

    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0

    print('running eval...')

    with torch.no_grad():
        for (imgs, img_segs, rots, trans, intrins, post_rots, post_trans, binimgs) in val_loader:

            if type == "seg":
                imgs = torch.cat((imgs, img_segs), 2)

            preds = model(imgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

            cv2.imwrite("pred_val_" + type + ".jpg", np.array(preds.sigmoid().detach().cpu())[0, 0] * 255)
            cv2.imwrite("binimgs_val_" + type + ".jpg", np.array(binimgs.detach().cpu())[0, 0] * 255)

    model.train()

    return {
        'loss': total_loss / len(val_loader.dataset),
        'iou': total_intersect / total_union,
    }
'''



#############################################################

def carla_dataloader(
        dataroot='../Downloads/carla',
        nepochs=10000,
        gpuid=0,

        H=128, W=352,
        resize_lim=(0.193, 0.225),
        final_dim=(128, 352),
        bot_pct_lim=(0.0, 0.22),
        rot_lim=(-5.4, 5.4),
        rand_flip=True,

        ncams=5,
        max_grad_norm=5.0,
        pos_weight=2.13,
        logdir='./runs',
        type='default',
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 45.0, 1.0],

        bsz=4,
        val_step=2000,
        nworkers=10,
        lr=1e-3,
        weight_decay=1e-7,
):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }

    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
     # 14980 1404
    train_ticks = 7594
    val_ticks = 1404

    train_dataset = CarlaDataset(os.path.join(dataroot, "train/"), data_aug_conf, train_ticks)
    val_dataset = CarlaDataset(os.path.join(dataroot, "val/"), data_aug_conf, val_ticks)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True,
                                               num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz,
                                             shuffle=False, num_workers=nworkers)

    return train_loader, val_loader

def nuscenes_dataloader(version='mini',
            dataroot='../Downloads/nuscenes',
            nepochs=10000,
            gpuid=1,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',
            type='default',
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=4,
            val_step=-1,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    return trainloader, valloader

###################
# 
# fdal demo .py
# 
#############

def scheduler(optimizer_, init_lr_, decay_step_, gamma_):
    class DecayLRAfter:
        def __init__(self, optimizer, init_lr, decay_step, gamma):
            self.init_lr = init_lr
            self.gamma = gamma
            self.optimizer = optimizer
            self.iter_num = 0
            self.decay_step = decay_step

        def get_lr(self) -> float:
            if ((self.iter_num + 1) % self.decay_step) == 0:
                lr = self.init_lr * self.gamma
                self.init_lr = lr

            return self.init_lr

        def step(self):
            """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                if 'lr_mult' not in param_group:
                    param_group['lr_mult'] = 1.
                param_group['lr'] = lr * param_group['lr_mult']

            self.iter_num += 1

        def __str__(self):
            return str(self.__dict__)

    return DecayLRAfter(optimizer_, init_lr_, decay_step_, gamma_)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False

def sample_batch(train_source, train_target, device, indx):

    (imgs_s, img_segs, rots_s, trans_s, intrins_s, post_rots_s, post_trans_s, binimgs_s) = next(train_source) #carla
    (imgs_t, rots_t, trans_t, intrins_t, post_rots_t, post_trans_t, binimgs_t) = next(train_target) #nuscenes
    
    imgs_s, rots_s, trans_s = imgs_s.to(device), rots_s.to(device), trans_s.to(device)
    intrins_s, post_rots_s, post_trans_s = intrins_s.to(device), post_rots_s.to(device), post_trans_s.to(device)
    
    imgs_t, rots_t, trans_t = imgs_t.to(device), rots_t.to(device), trans_t.to(device)
    intrins_t, post_rots_t, post_trans_t = intrins_t.to(device), post_rots_t.to(device), post_trans_t.to(device)
    
    binimgs_s = binimgs_s.to(device)

    X_s = (imgs_s, rots_s, trans_s, intrins_s, post_rots_s, post_trans_s)
    X_t = (imgs_t, rots_t, trans_t, intrins_t, post_rots_t, post_trans_t)

    return X_s, X_t, binimgs_s

#############################
#
# main training func
#
#

def main(divergence='pearson', n_epochs=30, iter_per_epoch=3000, lr=0.01, wd=0.002, reg_coef=0.5, seed=2):

    seed_all(seed)
    xbound=[-50.0, 50.0, 0.5]
    ybound=[-50.0, 50.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 45.0, 1.0]
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')

    #taskhead, aux_head, backbone = compile_model(grid_conf)
    model = compile_model(grid_conf)

    model.load_state_dict(torch.load('./checkpoint.pt'))
    model = model.to(device)

    num_classes = 2 # binary seg

    # load the dataloaders.
    train_source, val_source = carla_dataloader()
    train_source = itertools.cycle(train_source)
    train_target, test_loader = nuscenes_dataloader()
    train_target = itertools.cycle(train_target)

    # define the loss function....
    taskloss = nn.CrossEntropyLoss()
    taskloss = taskloss.to(device)

    # fDAL ----
    #train_target = ForeverDataIterator(train_target)
    #train_source = ForeverDataIterator(train_source)

    #learner = fDALLearner(backbone, taskhead, taskloss, divergence=divergence, reg_coef=reg_coef, n_classes=num_classes,
    #                      grl_params={"max_iters": 3000, "hi": 0.6, "auto_step": True}  # ignore for defaults.
    #                      )

    learner = fDALLearner(model, taskloss, divergence=divergence, reg_coef=reg_coef, n_classes = num_classes,
			   grl_params={"max_iters": 3000, "hi": 0.6, "auto_step": True})

    
    

    # define the optimizer.

    # Hyperparams and scheduler follows CDAN.
    opt = optim.SGD(learner.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    opt_schedule = scheduler(opt, lr, decay_step_=iter_per_epoch * 5, gamma_=0.5)

    #print(model.device,'model device')

    print('Starting training...')
    for epochs in range(n_epochs):
        learner.train()
        for i in range(iter_per_epoch):
            opt_schedule.step()
            # batch data loading...
            x_s, x_t, labels_s = sample_batch(train_source, train_target, device, i)
            # forward and loss
            loss, others = learner((x_s, x_t), labels_s)
            pred_s = others['pred_s']
            # opt stuff
            opt.zero_grad()
            loss.backward()
            # avoid gradient issues if any early on training.
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 10)
            opt.step()
            if i % 1500 == 0:
                _, _, iou = get_batch_iou(pred_s, labels_s)
                print("Epoch",epochs, "train iou:", iou)

        val_info = get_val_info(learner.get_reusable_model(True), test_loader, SimpleLoss, device)
        
        print(f"Epoch:{epochs} nuscenes loss: {val_info['loss']} nuscenes iou: {val_info['iou']}")

    # save the model.
    torch.save(learner.get_reusable_model(True).state_dict(), './checkpoint.pt')
    print('done.')

if __name__ == "__main__":
    fire.Fire(main)

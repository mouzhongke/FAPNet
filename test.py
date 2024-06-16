import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse


from lib.model import FAPNet
from utils.dataloader import tt_dataset
from utils.eva_funcs import eval_Smeasure,eval_mae,numpy2tensor
import scipy.io as scio 
import cv2
import imageio
from skimage import img_as_ubyte


parser = argparse.ArgumentParser()
parser.add_argument('--testsize',   type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str, default='/mnt/harddisk3/Pengxiaolong/codespace/FAPNet-main/output/FAPNet_best.pth')
parser.add_argument('--save_path',  type=str, default='/mnt/harddisk3/Pengxiaolong/codespace/FAPNet-main/result/')

opt   = parser.parse_args()
model = FAPNet(channel=64).cuda()


# cur_model_path = opt.model_path+'FAPNet.pth'
cur_model_path = opt.model_path
model.load_state_dict(torch.load(cur_model_path))
model.eval()
        
    
################################################################

for dataset in ['CHAMELEON', 'CAMO', 'COD10K ']:
    
    save_path = opt.save_path + dataset + '/'
    os.makedirs(save_path, exist_ok=True)        
        
        
    test_loader = tt_dataset('/mnt/datasets/COD/Dataset/TestDataset/CAMO/Imgs/'.format(dataset),
                               '/mnt/datasets/COD/Dataset/TestDataset/CAMO/GT/'.format(dataset), opt.testsize)

    img_count = 1
    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        cam, _1, _2, _3, _4 = model(image)
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        imageio.imsave(save_path + name, img_as_ubyte(cam))
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
        # _,_,_, cam,_ = model(image)
        #
        # res = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        
################################################################
        # cv2.imwrite(save_path+name, res*255)
        


 
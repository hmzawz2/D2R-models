import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch

import options as option
from models import create_model

import utils as util
from seamless_utils import *
import torch.nn.functional as F
from tifffile import imread, imwrite
from tqdm import tqdm
from einops import rearrange
from pathlib import Path

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default="./options/test/FIB_test.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

# os.system("rm ./result")
# os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

# util.setup_logger(
#     "base",
#     opt["path"]["log"],
#     "test_" + opt["name"],
#     level=logging.INFO,
#     screen=True,
#     tofile=True,
# )
# logger = logging.getLogger("base")
# logger.info(option.dict2str(opt))


# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)

sde.set_model(model.model)

scale = opt['degradation']['scale']
# opt["path"]["results_root"] = "/home/chenbh/python/EDiffSR-main/outputs"


###### Pred volume ######

def batch_pred(model, sde, volume, scale_factor, batch_size):
    d, _, _ = volume.shape
    vols = []
    for ind in range(0, d, batch_size):
        # crop as batch_size, 1, h, w
        slices_t = volume[ind:ind+batch_size].to(model.device)
        slices_t = rearrange(slices_t, '(n c) h w -> n c h w', c=1)
        slices_t = F.interpolate(slices_t, scale_factor=(scale_factor, 1), mode='bicubic')

        # inference
        noisy_state = sde.noise_state(slices_t)
        model.feed_data(noisy_state, slices_t)
        model.test(sde, save_states=False)
        sr_imgs = model.output.float().cpu().squeeze()
        vols.append(sr_imgs)

    vol = torch.cat(vols, dim=0)
    return vol

def run_model(
    model,
    sde,
    test_volume,
    scale_factor,
    image_shape=(64, 256, 256), # d h w
    overlap=(16, 16, 16),
    batch_size=64,
):
    # ----------
    #  Crop Subvolumes
    # ----------
    inpu_d, inpu_h, inpu_w = image_shape
    assert inpu_h % batch_size == 0 and inpu_w % batch_size == 0, "Input shape should change"

    coord_np, z_crop_num, y_crop_num, x_crop_num = create_coord(test_volume.shape, image_shape, overlap)
    test_shape = test_volume.shape
    vol_ls = []
    with torch.no_grad():
        for i in tqdm(range(coord_np.shape[1])):
            z, y, x = coord_np[0, i], coord_np[1, i], coord_np[2, i]
            crop = np.s_[
                z - image_shape[0] // 2 : z + image_shape[0] // 2,
                y - image_shape[1] // 2 : y + image_shape[1] // 2,
                x - image_shape[2] // 2 : x + image_shape[2] // 2,
            ]
            volume = test_volume[crop]
            volume = np.ascontiguousarray(volume).astype(np.float32) / 255.0
            volume = torch.from_numpy(volume)

            # h pred
            volume_h = rearrange(volume, 'd h w -> h d w')
            pred_h = batch_pred(model, sde, volume_h, scale_factor, batch_size)
            pred_h = rearrange(pred_h, 'h d w -> d h w').detach().cpu().numpy()
            
            # w pred
            volume_w = rearrange(volume, 'd h w -> w d h')
            pred_w = batch_pred(model, sde, volume_w, scale_factor, batch_size)
            pred_w = rearrange(pred_w, 'w d h -> d h w').detach().cpu().numpy()
            
            # average
            pred = (pred_h + pred_w) / 2.0
            vol_ls.append(pred)

            imwrite(f"block_log/{i}.tif", pred)

    # ----------
    #  3D Stitch
    # ----------
    res = blend_volume(vol_ls, z_crop_num, y_crop_num, x_crop_num, test_shape, scale_factor, overlap=overlap)
    res = (np.clip(res, 0, 1) * 255.0).astype("uint8")
    return res

volume = imread("Gour21_11x11x30nm-crop_537_245_138_inpu.tif")
vol = run_model(model, sde, volume, 4, image_shape=(64, 256, 256), overlap=(4, 4, 4), batch_size=32)
imwrite("gour_IRSDE_x4.tif", vol)

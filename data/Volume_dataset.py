from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data as data
from tifffile import imread
from einops import rearrange

class VolumeDataset(data.Dataset):
    """
    Read Volume (High Resolution in XY-plane, refered as HR) and randomly crop with downsample.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stack_list = []

        assert self.opt["LR_size"] * self.opt["scale"] == self.opt["GT_size"], "Downsample ratio error."
        
        print("\033[1;31mImage list -----> \033[0m")
        print("All files are in -----> ", opt["dataroot_GT"])
        if Path(opt["dataroot_GT"]).is_file():
            self.stack_name_list = [opt["dataroot_GT"]]
        else:
            self.stack_name_list = list(Path(opt["dataroot_GT"]).glob("*.tif"))
        print("Total stack number -----> ", len(self.stack_name_list))
        print("Stack name lists:")
        for name in self.stack_name_list:
            print(f"\t{Path(name).name}")
            self.stack_list.append(imread(name))


    def __getitem__(self, index):
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        volume_ind = np.random.randint(0, len(self.stack_name_list))
        depth, height, width = self.stack_list[volume_ind].shape
        sd = np.random.randint(0, depth)
        sh = np.random.randint(0, height - GT_size)
        sw = np.random.randint(0, width - GT_size)
        gt_slice = self.stack_list[volume_ind][sd:sd+1, sh:sh+GT_size, sw:sw+GT_size]
        
        if self.opt["phase"] == "train":
            if random.random() >= 0.5:
                gt_slice = gt_slice[:, ::-1]
            if random.random() >= 0.5:
                gt_slice = gt_slice[:, :, ::-1]
            rotations = random.choice([0, 1, 2, 3])
            gt_slice = np.rot90(gt_slice, rotations, axes=(1, 2))

        lr_slice = gt_slice[:, ::scale].copy()
        gt_slice, lr_slice = gt_slice.astype(np.float32) / 255.0, lr_slice.astype(np.float32) / 255.0
        
        img_GT = torch.from_numpy(np.ascontiguousarray(gt_slice)).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(lr_slice)).float()

        return {"LQ": img_LR, "GT": img_GT}
    
    def __len__(self):
        if self.opt["phase"] == "train":
            return 540
        return 32

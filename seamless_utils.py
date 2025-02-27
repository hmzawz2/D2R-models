import itertools
import math
import torch
import numpy as np

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def rotate_8(imgs_hr):
    alpha_ls = [-2,0]
    beta_ls = [-2]
    gamma_ls = [-2, -1,0,1]

    imgs_hr_ls=[]
    idx=-1
    for alpha in alpha_ls:
        for beta in beta_ls:
            for gamma in gamma_ls:
                idx+=1
                imgs_hr_copy = imgs_hr.clone()
                imgs_hr_copy=torch.rot90(imgs_hr_copy, 2+alpha, [2, 4])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + beta, [2, 3])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 2 + gamma, [3, 4])
                imgs_hr_ls.append(imgs_hr_copy)
    return imgs_hr_ls


def anti_rotate_8(imgs_hr_ls,image_shape=(128,128,128)):
    alpha_ls = [-2, 0]
    beta_ls = [-2]
    gamma_ls = [-2, -1, 0, 1]

    idx=-1
    imgs_anti_rot =[]
    res=torch.zeros((1,1,image_shape[0],image_shape[1],image_shape[2])).type(Tensor)
    for alpha in alpha_ls:
        for beta in beta_ls:
            for gamma in gamma_ls:
                idx+=1
                imgs_hr_copy = imgs_hr_ls[idx]
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 4-(2 +gamma), [3, 4])
                imgs_hr_copy = torch.rot90(imgs_hr_copy, 4-(2 + beta), [2, 3])
                imgs_hr_copy=torch.rot90(imgs_hr_copy, 4-(2+alpha), [2, 4])
                imgs_anti_rot.append(imgs_hr_copy)
                res+=imgs_hr_copy

    return imgs_anti_rot,res/8

def blend_X(img1, img2, overlap):
    b, a = img1.shape
    d, c = img2.shape
    res = np.zeros([b, a+c - overlap], dtype=np.float16)

    weights1 = np.expand_dims(np.linspace(1, 0, overlap, dtype=np.float16), axis=0)
    weights2 = np.expand_dims(np.linspace(0, 1, overlap, dtype=np.float16), axis=0)

    res[:, :a-overlap] = img1[:, :a-overlap]
    res[:, a-overlap:a] = img1[:, a-overlap:] * weights1 + img2[:, :overlap] * weights2
    res[:, a:] = img2[:, overlap:]

    return res

def blend_Y(img1, img2, overlap):
    b, a = img1.shape
    d, c = img2.shape
    res = np.zeros([b+d - overlap, a], dtype=np.float16)

    weights1 = np.expand_dims(np.linspace(1, 0, overlap, dtype=np.float16), axis=1)
    weights2 = np.expand_dims(np.linspace(0, 1, overlap, dtype=np.float16), axis=1)

    res[:b-overlap, :] = img1[:b-overlap, :]
    res[b-overlap:b, :] = img1[b-overlap:, :] * weights1 + img2[:overlap, :] * weights2
    res[b:, :] = img2[overlap:, :]

    return res

def blend3D_X(img1, img2, overlap=16):
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c, b, a+x - overlap], dtype=np.float16)
    for j in range(b):
        res[:, j, :] = blend_X(img1[:, j, :], img2[:, j, :], overlap=overlap)
    return res

def blend3D_Y(img1, img2, overlap=16):
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c, b + y - overlap, a], dtype=np.float16)
    for j in range(a):
        res[:, :, j] = blend_X(img1[:, :, j], img2[:, :, j], overlap=overlap)
    return res

def blend3D_Z(img1, img2, overlap=16):
    c, b, a = img1.shape
    z, y, x = img2.shape
    res = np.zeros([c + z - overlap, b, a], dtype=np.float16)
    for j in range(b):
        res[:, j, :] = blend_Y(img1[:, j, :], img2[:, j, :], overlap=overlap)
    return res

def blend_volume(vol_ls, z_crop_num, y_crop_num, x_crop_num, test_shape, scale_factor, overlap=(8, 8, 8)):

    x_ls = [None] * (z_crop_num * y_crop_num)
    for i in range(z_crop_num):
        for j in range(y_crop_num):
            x_temp = None
            for k in range(x_crop_num):
                nps = i * y_crop_num * x_crop_num + j * x_crop_num + k
                vol_nps = vol_ls[nps]
                if x_temp is not None:
                    ovlp = overlap[2] if x_temp.shape[2] + vol_nps.shape[2] - overlap[2] <= test_shape[2] else x_temp.shape[2] + vol_nps.shape[2] - test_shape[2]
                    x_temp = blend3D_X(x_temp, vol_nps, overlap=ovlp)
                else:
                    x_temp = vol_nps
            x_ls[i * y_crop_num + j] = x_temp
    print("Blend X done.")

    y_ls = [None] * z_crop_num
    for i in range(z_crop_num):
        y_temp = None
        for j in range(y_crop_num):
            nps = i * y_crop_num + j
            x_nps = x_ls[nps]
            if y_temp is not None:
                ovlp = overlap[1] if y_temp.shape[1] + x_nps.shape[1] - overlap[1] <= test_shape[1] else y_temp.shape[1] + x_nps.shape[1] - test_shape[1]
                y_temp = blend3D_Y(y_temp, x_nps, overlap=ovlp)
            else:
                y_temp = x_nps
        y_ls[i] = y_temp
    print("Blend Y done.")

    z_temp = None
    for i in range(z_crop_num):
        if z_temp is not None:
            ovlp = int(overlap[0] * scale_factor) if z_temp.shape[0] + y_ls[i].shape[0] - int(overlap[0] * scale_factor) <= int(test_shape[0] * scale_factor) else z_temp.shape[0] + y_ls[i].shape[0] - int(test_shape[0] * scale_factor)
            z_temp = blend3D_Z(z_temp, y_ls[i], overlap=ovlp)
        else:
            z_temp = y_ls[i]
    print("Blend Z done.")

    return z_temp


def get_crop_num(length, crop_size=128, overlap=16):
    """length=n*crop_size-(n-1)*overlap"""
    num = (length - overlap) / (crop_size - overlap)
    return math.ceil(num)


def create_coord(s1, s2=(128, 128, 128), overlap=(16, 16, 16)):
    coord_ls = [[], [], []]
    z_crop_num = get_crop_num(s1[0], crop_size=s2[0], overlap=overlap[0])
    y_crop_num = get_crop_num(s1[1], crop_size=s2[1], overlap=overlap[1])
    x_crop_num = get_crop_num(s1[2], crop_size=s2[2], overlap=overlap[2])

    for z, y, x in itertools.product(
        range(z_crop_num), range(y_crop_num), range(x_crop_num)
    ):
        z_coord = (
            s2[0] // 2 + z * (s2[0] - overlap[0])
            if (s2[0] // 2 + z * (s2[0] - overlap[0])) < s1[0] - s2[0] // 2
            else s1[0] - s2[0] // 2
        )
        y_coord = (
            s2[1] // 2 + y * (s2[1] - overlap[1])
            if (s2[1] // 2 + y * (s2[1] - overlap[1])) < s1[1] - s2[1] // 2
            else s1[1] - s2[1] // 2
        )
        x_coord = (
            s2[2] // 2 + x * (s2[2] - overlap[2])
            if (s2[2] // 2 + x * (s2[2] - overlap[2])) < s1[2] - s2[2] // 2
            else s1[2] - s2[2] // 2
        )
        coord_ls[0].append(z_coord)
        coord_ls[1].append(y_coord)
        coord_ls[2].append(x_coord)

    return np.array(coord_ls), z_crop_num, y_crop_num, x_crop_num
# D2R Networks

This is the resp of D2R (Diffusion to Resolution) training framework.

D2R is a training framework of volume microscopy, which uses diffusion model to restore lateral slices of volume as pseudo-high-resolution (pseudo-HR) volumes to train 3D volume super-resolution (VSR) network.

## D2R Step1&2

In step 1&2, D2R need to train a diffusion model to restore degraded slice. We use a light-weight version implementation of IRSDE to learn how to restore slices. You can use your own data to train one channel data restoration model, and generate pseudo-HR volumes as follows:

1. Write your own data description file, and save in 'options' folder.
2. Use 'train.py' to start training.
3. Run the training script until you find it converage. Move your model to 'pretrained' folder.
4. Use 'volume_infer.py' to generate pseudo-HR volume. You need to point out which model you want to use and your low-resolution volume.

After that, you will have finished D2R step1&2.

## D2R Step3

You can try to train your own VSR model with pseudo-HR training data. Besides, we also provide 3DSRUNet in our resp. The Axial Enhancement Network (AENet) will be coming soon.

import random
import os
import numpy as np
import h5py
import cv2
import math
import torch
import PIL
import torch.nn as nn
from torchvision import transforms
import glob
import warnings
import torchvision.transforms.functional as F

def load_data(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


    img_shape=img.shape
    scale=(0.08,1.0)
    ratio=(3./4.,4./3.)
    
    heatmap_path = img_path.replace("images","hmap").replace('.jpg','.npy')
    size_path = img_path.replace("images","size").replace('.jpg','.npy')
    offset_path = img_path.replace("images","offset").replace('.jpg','.npy')


    # image preprocessing and augmentation

    hmap_file=np.load(heatmap_path)
    offset_file=np.load(offset_path)
    size_file=np.load(size_path)
  

   

    height,width,_=img.shape
    area=width*height

    random_param=()    #store random generated numbers
    for attempt in range(250):
        target_area=np.random.uniform(*scale)*area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w=min(int(round(math.sqrt(target_area*aspect_ratio))),800)
        h=min(int(round(math.sqrt(target_area/aspect_ratio))),800)

        if 0 < w <= width and 0 < h <= height:
            i=np.random.randint(0,height-h)
            j=np.random.randint(0,width-w)
            random_param=(i,j,h,w)
            break



    i,j,h,w=random_param
    annot_i,annot_j,annot_h,annot_w=(x//4 for x in random_param)

    image_interpolation=PIL.Image.BILINEAR
    annot_interpolation=PIL.Image.NEAREST

    img=np.array(F.resized_crop(PIL.Image.fromarray(img),i,j,h,w,(800,800),image_interpolation))
    hmap_img=np.array(F.resized_crop(PIL.Image.fromarray(hmap_file),annot_i,annot_j,annot_h,annot_w,(200,200),annot_interpolation))
    offset=np.array(F.resized_crop(PIL.Image.fromarray(np.uint8(offset_file),mode='LA'),annot_i,annot_j,annot_h,annot_w,(200,200),annot_interpolation))
    size_img=np.array(F.resized_crop(PIL.Image.fromarray(np.uint8(size_file),mode='LA'),annot_i,annot_j,annot_h,annot_w,(200,200),annot_interpolation))

#--------------------------------------done cropping------------------------------------------------------------

#--------------------------------------upscaling--------------------------------------------------------------



    
    # upsample_img=nn.Upsample(size=(800,800),mode='bilinear')
    # upsample_annot=nn.Upsample(size=(200,200),mode='nearest')

    # img=upsample_img(torch.tensor(img))
    # hmap=upsample_annot(torch.tensor(hmap))
    # offset=upsample_annot(torch.tensor(offset))
    # size_img=upsample_annot(torch.tensor(size_img))




    return np.array(img),np.array(hmap_img),np.array(offset),np.array(size_img)


#     image_shape=img.shape
#     height=image_shape[1]
#     width=image_shape[0]

#     image,target=image_transforms(image,target)


#     if(not full_img):
#         image_shape = img.shape
#         height, width = False, False

#         if image_shape[0]<512 or image_shape[1]<512:
#             if image_shape[0]<image_shape[1]:
#                 height=True
#             else:
#                 width=True

#         if height:
#             new_height = 512
#             ratio = new_height/image_shape[0]
#             new_width = int(ratio*image_shape[1])
#             img, target, target_at = reshape_data(new_width, new_height, img, target, target_at)
#             # rand_scale=random.randint(8,12)/10
#             # (height_scaled,width_scaled,_)=img.shape
#             # img, target, target_at = reshape_data(int(rand_scale*width_scaled),int(rand_scale*height_scaled), img, target, target_at)


#         if width:
#             new_width = 512
#             ratio = new_width/image_shape[1]
#             new_height = int(ratio*image_shape[0])
#             img, target, target_at = reshape_data(new_width, new_height, img, target, target_at)
#             # rand_scale=random.randint(8,12)/10
#             # (height_scaled,width_scaled,_)=img.shape
#             # img, target, target_at = reshape_data(int(rand_scale*width_scaled),int(rand_scale*height_scaled), img, target, target_at)


#         # random crop of 400x400 image

#         h,w = img.shape[0:2]
#         th, tw = (400,400)
#         i = random.randint(0, h-th)
#         j = random.randint(0, w-tw)
#         # print(img.shape,target_at.shape,target.shape)
#         # print(i,j,th,tw,h,w)
#         img = img[i:i+th, j:j+tw, :]
#         target = target[i:i+th, j:j+tw]
#         target_at = target_at[i:i+th, j:j+tw]
#         # print(img.shape,target_at.shape,target.shape)


#     # data augmentation random gray scaling
#     if(train):
#         # if random.random()<0.1:
#         #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         #     img = np.expand_dims(img, 2).repeat(3,axis=2)

#         # data augmentation random flip
#         if random.random()<0.5:
#             target = np.fliplr(target)
#             target_at = np.fliplr(target_at)
#             img = cv2.flip(img, 1)
            
#         # if random.random()<0.3:
#         #     img = adjust_gamma(img, gamma=random.uniform(0.5,1.5))

#     # resizing image and target in powers of 2, IF using deconv or upsampling

#     # INTER_LINEAR # INTER_CUBIC # INTER_NEAREST # INTER_AREA (when downsampling)
#     if max_down:
#         # print()
#         # print(img.shape,target_at.shape,target.shape)
#         assert img.shape[0]==target.shape[0]==target_at.shape[0] and img.shape[1]==target.shape[1]==target_at.shape[1]

#         new_shape = ((img.shape[0])//max_down)*max_down, ((img.shape[1])//max_down)*max_down
#         img, target, target_at = reshape_data(new_shape[1], new_shape[0], img, target, target_at)


#     # downscaling output
#     target = cv2.resize(target, (target.shape[1]//output_down,target.shape[0]//output_down), interpolation=cv2.INTER_LINEAR)*(output_down**2)
#     target_at = cv2.resize(target_at, (target_at.shape[1]//output_down,target_at.shape[0]//output_down), interpolation=cv2.INTER_NEAREST)
    
#     # converting data into tensors in the right format
#     img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255)
#     target = torch.from_numpy(target).unsqueeze(0)
#     target_at = torch.from_numpy(target_at).unsqueeze(0)

#     return img,target,target_at


# def reshape_data(width, height, img, target, target_at):
#     img = cv2.resize(img, ((width),(height)), interpolation=cv2.INTER_LINEAR)
#     target = cv2.resize(target, ((width),(height)), interpolation=cv2.INTER_LINEAR)*((width/target.shape[1])*(height/target.shape[0]))
#     target_at = cv2.resize(target_at, ((width),(height)), interpolation=cv2.INTER_NEAREST)
#     return img, target, target_at


# def adjust_gamma(image, gamma=1.0, gain=1):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255 * gain
#         for i in np.arange(0, 256)]).astype("uint8")
#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)


# def image_transforms(image):
#     transforms=transforms.Compose([
#                             transforms.RandomHorizontalFlip(0.5),
#                             transforms.RandomResizedCrop(800,
#                                 scale=(0.8,1.0),
#                                 ratio=(0.75,1.333333333),
#                                 interpolation=PIL.Image.INTER_NEAREST
#                                 )
#     ])

#     image=transforms(image)
#     return image


# def RandomResizeCrop(img,random_param,size,scale,ratio,interpolation):
    

    

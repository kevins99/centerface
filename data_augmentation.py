import numpy as np
import cv2
import torch
import random
import torchvision
import json


# def check_face(annots, crop_im,image):
#     for annot in annots:
#         x1 = int(annot[0])
#         x2 =x1 + int(annot[2])
#         y1 = int(annot[1])
#         y2 = y1 + int(annot[3])
#         print(x1,y1,x2,y2)
#         print((x2-x1)*(y2-y1),0.3*(image.shape[0]*image.shape[1]))
#         if(x1<0 or x2>image.shape[1] or y1<0 or y2>image.shape[0] or (x2-x1)*(y2-y1)>0.3*(image.shape[0]*image.shape[1])):
#             return True
#         # print(x1)
#         elif crop_im[0] <= x1 and crop_im[2] >= x2 and crop_im[1] <= y1 and crop_im[3] >= y2:
#             return True 
#         else:
#             return False


def crop(img_path):

    threshold = 0.5
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    with open('annots.json', 'rb') as f:
        dict_annots = json.load(f)
    
    annots = dict_annots.get(img_path.split('/')[-1])
    # print(annots)

    heatmap_path = img_path.replace("images", "hmaps").replace("jpg", "npy")
    size_path = img_path.replace("images", "sizes").replace("jpg", "npy")
    offset_path = img_path.replace("images", "offsets").replace("jpg", "npy")
    # print(img_path.replace("images", "hmaps").replace("jpg", "npy"))
    # changed directory names for hmap to hmaps


    height, width, _ = image.shape
    # print("height image",height)
    # print("width image",width)

    heatmap = np.load(heatmap_path)
    heatmap=np.clip(np.round(heatmap),1e-9,0.9999999)
    size = np.load(size_path)
    offset = np.load(offset_path)

    # print("OFFSET POST load ",offset.shape) #DEBUG OFFSET SHAPE

    SCALE =  [1.,0.9,0.75,0.6,0.5,0.55,0.65,0.7,0.4,0.45,0.3,0.35,0.2,0.25,0.15,0.8]#0.75,0.6,0.5,0.55,0.65,0.7,0.4,0.45,0.3,0.35,0.2,0.25,0.15,0.8#[1.]#np.linspace(0.08,1,92)
    for _ in range(1000):
        
        scale = random.choice(SCALE)

        short_side = min(height, width)
        w = min(int(scale*short_side) - int(scale*short_side)%4,800)
        # h = w = 800
        h = w



        l = random.randrange(0,width-w + 1)
        left = l - (l%4)
        # print("LEFT ",left)
        # print(f"exrtra height:{height-h}")
        t = random.randrange(0,height-h + 1) # changed height-t to height-h
        top = t - (t%4)

        # print("TOP ",top)

        # put in if else for cases where w==widht or h==height because randrange was 0
        # left = random.randrange(width-w)
        # top = random.randrange(height-h)

        crop_rgn = [left, top, left+w, top+h]
        crop_im = image[crop_rgn[1]:crop_rgn[3], crop_rgn[0]:crop_rgn[2]]

        heatmap = heatmap[int(crop_rgn[1]/4):int(crop_rgn[3]/4), int(crop_rgn[0]/4):int(crop_rgn[2]/4)]
        # cv2.imshow("qwe",heatmap*255**2)
        # cv2.waitKey(0)
        # rounded off all attributes
        c0, c1 = np.where(heatmap>=0.3)
        # print(c0,c1)

        # print(heatmap.shape,len(c0))
        if len(c0) >= 1:
        # print(np.sum(heatmap))
        # if check_face(annots, crop_rgn,image):


            size_img = size[:, int(crop_rgn[1]/4):int(crop_rgn[3]/4), int(crop_rgn[0]/4):int(crop_rgn[2]/4)]
            offset = offset[:, int(crop_rgn[1]/4):int(crop_rgn[3]/4), int(crop_rgn[0]/4):int(crop_rgn[2]/4)]



            offset[0, :, :] = offset[0, :, :]*(800//(scale*width))
            offset[1, :, :] = offset[1, :, :]*(800//(scale*height))

            size_img[0, :, :] = size_img[0, :, :]*(800//(scale*width))
            size_img[1, :, :] = size_img[1, :, :]*(800//(scale*height))

            # heatmap = np.transpose(heatmap,(1,0))
            offset = np.transpose(offset,(1,2,0))
            size_img = np.transpose(size_img,(1,2,0))

        # scale
            # print(hmap_img.shape)

            crop_im = cv2.resize(crop_im.astype(np.float32),(800,800),cv2.INTER_LINEAR)
            hmap_img = cv2.resize(heatmap.astype(np.float32),(200,200),cv2.INTER_NEAREST)
            offset = cv2.resize(offset,(200,200),cv2.INTER_NEAREST)
            size_img = cv2.resize(size_img,(200,200),cv2.INTER_NEAREST)

            # hmap_img = np.transpose(hmap_img,(0,1))
            offset = np.transpose(offset,(2,0,1))
            size_img = np.transpose(size_img,(2,0,1))

            # print(crop_im.shape, hmap_img.shape, offset.shape, size_img.shape)
            # cv2.imshow("hmap",hmap_img*255*255)
            # cv2.imshow("img",crop_im.astype(np.uint8))
            # cv2.waitKey(0)

            return crop_im, hmap_img, offset, size_img
        else:
            # print()
            heatmap = np.load(heatmap_path)
            heatmap[np.where(heatmap>0)]=1

            # heatmap=np.clip(np.round(heatmap),1e-9,0.9999999)
            scale = np.load(size_path)
            offset = np.load(offset_path)
            # count = count + 1
            continue

    test=np.zeros_like(image)
    print(test.shape)
    test=cv2.resize(test,(image.shape[1]//4,image.shape[0]//4))
    print(test.shape)
    test[:,:,0]=heatmap[np.where(heatmap>0)]=1
    test[:,:,1]=test[:,:,0]
    test[:,:,2]=test[:,:,0]

    cv2.imshow("test!!!",test)
    cv2.waitKey(0)
        # print(count)


def flip(image, heatmap, offset, scale):
    
    # image = cv2.imread(img_path)
    
    # heatmap_path = img_path.replace("images", "hmap").replace("jpg", "npy")
    # size_path = img_path.replace("images", "size").replace("jpg", "npy")
    # offset_path = img_path.replace("images", "offset").replace("jpg", "npy")

    # heatmap = np.load(heatmap_path)
    # scale = np.load(size_path)
    # offset = np.load(offset_path)
    # print(f"image: {image.shape}, heatmap{heatmap.shape}, offset:{offset.shape}, size:{scale.shape}")
    
    image = np.fliplr(image)
    heatmap = np.fliplr(heatmap)
    scale = np.fliplr(scale)
    offset = np.fliplr(offset)
    offset[0] =  -1*offset[0]

    return image, heatmap, offset, scale


def distort(image):

    
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def load_data(img_path, train=True):
    # print(img_path)
    if not train:
        
        heatmap_path = img_path.replace("images", "hmaps").replace("jpg", "npy")
        size_path = img_path.replace("images", "sizes").replace("jpg", "npy")
        offset_path = img_path.replace("images", "offsets").replace("jpg", "npy")
        heatmap = np.load(heatmap_path)
        scale = np.load(size_path)
        offset = np.load(offset_path)

        return image, heatmap, offset, scale


    if train:
        chance = np.random.random()
        image, heatmap,offset,scale = crop(img_path) #offset, scale
        if chance < 0.0:     
            image, heatmap, offset, scale = flip(image, heatmap, offset, scale)
        
        if chance < 0.00:
            image = distort(image)

        return image, heatmap, offset, scale

    # heatmap = np.transpose(heatmap, (1, 2, 0))
    # offset = np.transpose(offset, (1, 2, 0))
    # scale = np.transpose(scale, (1, 2, 0))

    


    

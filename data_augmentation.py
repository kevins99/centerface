import numpy as np
import cv2
import torch
import random
import torchvision

def crop(img_path):

    threshold = 0.5
    image = cv2.imread(img_path)
    
    heatmap_path = img_path.replace("images", "hmaps").replace("jpg", "npy")
    size_path = img_path.replace("images", "sizes").replace("jpg", "npy")
    offset_path = img_path.replace("images", "offsets").replace("jpg", "npy")
    # print(img_path.replace("images", "hmaps").replace("jpg", "npy"))
    # changed directory names for hmap to hmaps

    height, width, _ = image.shape
    # print("height image",height)
    # print("width image",width)

    heatmap = np.load(heatmap_path)
    scale = np.load(size_path)
    offset = np.load(offset_path)

    # print("OFFSET POST load ",offset.shape) #DEBUG OFFSET SHAPE

    for _ in range(1000):
        
        # SCALE = [0.7, 0.8, 0.9, 1]
        # scale = random.choice(SCALE)

        # short_side = min(height, width)
        # w = int(scale*short_side) - int(scale*short_side)%4
        h = w = 800

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
        c0, c1 = np.where(heatmap==1)
        # print(c0,c1)
        if len(c0) > 1:
            scale = scale[:, int(crop_rgn[1]/4):int(crop_rgn[3]/4), int(crop_rgn[0]/4):int(crop_rgn[2]/4)]
            offset = offset[:, int(crop_rgn[1]/4):int(crop_rgn[3]/4), int(crop_rgn[0]/4):int(crop_rgn[2]/4)]
            return crop_im, heatmap, offset, scale
        else:
            heatmap = np.load(heatmap_path)
            scale = np.load(size_path)
            offset = np.load(offset_path)
            # count = count + 1
            continue
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
        image, heatmap, offset, scale = crop(img_path)

        if chance < 0.5:     
            image, heatmap, offset, scale = flip(image, heatmap, offset, scale)
        
        if chance < 0.08:
            image = distort(image)

        return image, heatmap, offset, scale

    # heatmap = np.transpose(heatmap, (1, 2, 0))
    # offset = np.transpose(offset, (1, 2, 0))
    # scale = np.transpose(scale, (1, 2, 0))

    


    

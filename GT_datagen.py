# %%
import cv2
import os
import numpy as np
from tqdm import tqdm
# from centerface import CenterFace
from config import __C as cfg
import scipy
from scipy.ndimage import gaussian_filter
# %%
#cv2 resize
# center /3?
# make hmap every face?

class DataGen:
    def __init__(self, bbx_txt=cfg.PATH.ANNOT, target_root=cfg.PATH.ROOT, img_dir_struct="flat", scale_imgs=False):
        self.bbx_txt = bbx_txt
        self.img_root = os.path.join(target_root, "images")
        self.offset_root = os.path.join(target_root, "offsets")
        self.heatmap_root = os.path.join(target_root, "hmaps")
        self.sizes_root = os.path.join(target_root, "sizes")
        self.scale_imgs = scale_imgs
        # if all the images are in a single folder instead of flat directory structure
        self.img_dir_struct = img_dir_struct
        self.img_annots = {}

    def read_img(self, name):
        img = cv2.imread(os.path.join(self.img_root, name))
        scale = 1
        if self.scale_imgs:
            smaller_dim = min(img.shape[0], img.shape[1])
            if smaller_dim < 800:
                scale = int(np.ceil((800/smaller_dim)))
                img = cv2.resize(img, (img.shape[0]*scale, img.shape[1]*scale))
                assert img.shape[0] >= 800 and img.shape[1] >= 800
        return img, scale

    def generate_heatmaps(self, img, annot):

        heatmap = np.zeros((img.shape[:2]))
        heatmap[annot[5], annot[4]] = 1
        return heatmap

    def generate_annotations(self, name, shape, annots):
        offsets_x = np.zeros(shape)
        offsets_y = np.zeros(shape)
        box_sizes_x = np.zeros(shape)
        box_sizes_y = np.zeros(shape)

        for annot in annots:
            center = [int(np.floor(annot[4]/4)), int(np.floor(annot[5]/4))]
            print(center)
            offsets_x[center[1], center[0]] = annot[4]/4 - np.floor(annot[4]/4)
            offsets_y[center[1], center[0]] = annot[5]/4 - np.floor(annot[5]/4)
            # box_sizes_x[center[1], center[0]] = np.log(annot[2]/4)
            box_sizes_x[center[1], center[0]] = annot[2]/4
            box_sizes_x[center[1], center[0]] = np.log(annot[2]/4)

            box_sizes_y[center[1], center[0]] = np.log(annot[3]/4)

        offsets_x = np.expand_dims(offsets_x, axis=0)
        offsets_y = np.expand_dims(offsets_y, axis=0)
        box_sizes_x = np.expand_dims(box_sizes_x, axis=0)
        box_sizes_y = np.expand_dims(box_sizes_y, axis=0)

        offsets = np.concatenate((offsets_x, offsets_y), axis=0)
        box_sizes = np.concatenate((box_sizes_x, box_sizes_y), axis=0)
        assert offsets.shape == (2, shape[0], shape[1])
        assert box_sizes.shape == (2, shape[0], shape[1])
        np.save(os.path.join(self.offset_root, name), offsets)
        np.save(os.path.join(self.sizes_root, name), box_sizes)

    def generate_cmap(self, img):
        shape = img.shape
        # img = cv2.GaussianBlur(img, (3,3), 2)
        img = gaussian_filter(img, 3)

        print(np.sum(img))
        img = (img > 0.001).astype(np.int32)
        cv2.imshow("test", img*255*255)
        print(img.shape,"cmap")
        cv2.waitKey(0)
        assert img.shape == shape
        assert np.amin(img) == 0
        assert np.amax(img) == 1
        return img

    def downsample(self, img):
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4),
                   interpolation=cv2.INTER_NEAREST)

        return img

    def main(self):
        img_annots = {}
        with open(self.bbx_txt, 'r+') as gt_file:
            name = gt_file.readline().strip()
            if self.img_dir_struct == "flat":
                name = name.split("/")[1]
            while name:
                img, scale = self.read_img(name)
                img_annots[name] = []
                count = int(gt_file.readline().strip())
                while count:
                    print(f"processing image {name}....")
                    annots = gt_file.readline().strip().split()
                    annots = list(map(lambda x: int(x)*scale, annots))
                    annots = annots[:4]
                    # remove faces with pixel boxes smaller than 16 pixels in area
                    if annots[2]*annots[3] < 16:
                        count -= 1
                        continue
                    center = [int(annots[0]+annots[2]/2),  # round(x1+w/2)
                              int(annots[1]+annots[3]/2)]  # round(y1+w/2)
                    annots.extend(center)
                    # so now annots are [x1, y1, box_size_x, box_size_y, center_x,center_y]
                    img_annots[name].append(annots)
                    count -= 1
                    heatmap = self.generate_heatmaps(img, annots)
                    classification_map = self.generate_cmap(heatmap)
                    downsampled = self.downsample(classification_map)
                    print(downsampled.shape)
                    np.save(os.path.join(self.heatmap_root, name[:-4]),
                            downsampled)
                    break
                    img_annots[name].append(annots)
                self.generate_annotations(
                        name[:-4], downsampled.shape, img_annots[name])
                name = gt_file.readline().strip()
                if self.img_dir_struct == "flat":
                    name = name.split("/")[1]


if __name__ == "__main__":
    datagen = DataGen(img_dir_struct="flat")
    datagen.main()

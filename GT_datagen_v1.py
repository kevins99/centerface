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


class DataGen:
    def __init__(self, bbx_txt=cfg.PATH.ANNOT, target_root=cfg.PATH.ROOT, img_dir_struct="flat", scale_imgs=True):
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
        print(name)
        img = cv2.imread(os.path.join(self.img_root, name))
        scale = 1
        if self.scale_imgs:
            smaller_dim = min(img.shape[0], img.shape[1])
            if smaller_dim < 800:
                scale = 800 / img.shape[0]
                img = cv2.resize(
                    img, (round(img.shape[1]*scale), round(img.shape[0]*scale)))
                print(
                    f"resized image shape: {img.shape}, resizing scale:{scale}")
                assert img.shape[0]*scale >= 800
                assert img.shape[1]*scale >= 800
        cv2.imwrite(os.path.join(self.img_root,name),img)
        return img, scale

    def generate_heatmaps(self, img, annots):
        heatmap = np.zeros((img.shape[:2]))
        for annot in annots:
            heatmap[annot[5], annot[4]] = 1
        return heatmap

    def generate_annotations(self, name, shape, annots):
        offsets_x = np.zeros(shape)
        offsets_y = np.zeros(shape)
        box_sizes_x = np.zeros(shape)
        box_sizes_y = np.zeros(shape)

        for annot in annots:
            annot = list(map(lambda x: x/4, annot))
            center = [int(np.floor(annot[4])), int(np.floor(annot[5]))]
            offsets_x[center[1], center[0]] = annot[4] - np.floor(annot[4])
            offsets_y[center[1], center[0]] = annot[5] - np.floor(annot[5])
            box_sizes_x[center[1], center[0]] = np.log(annot[2])
            box_sizes_y[center[1], center[0]] = np.log(annot[3])

        offsets_x = np.expand_dims(offsets_x, axis=0)
        offsets_y = np.expand_dims(offsets_y, axis=0)
        box_sizes_x = np.expand_dims(box_sizes_x, axis=0)
        box_sizes_y = np.expand_dims(box_sizes_y, axis=0)

        # x1, x2 = np.where(offsets_x>0)
        # # y1, y2 = np.where(offsets_y>0)

        # n = zip(x1, x2)
        
        # for i in n:
        #     if

        offsets = np.concatenate((offsets_x, offsets_y), axis=0)
        box_sizes = np.concatenate((box_sizes_x, box_sizes_y), axis=0)
        # print(f"offset shape: {offsets.shape}, sizes shape:{box_sizes.shape}")
        # assert offsets.shape == (2, shape[0], shape[1]) and box_sizes.shape == (
        #     2, shape[0], shape[1])
        # assert offsets.shape[1] >= 200 and offsets.shape[2] >= 200 and box_sizes.shape[1] >= 200 and box_sizes.shape[2] >= 200

        
        # np.save(os.path.join(self.sizes_root, name), box_sizes)

        return offsets,box_sizes

    def generate_cmap(self, img):
        shape = img.shape
        # sigma 4 lead to really massive dots
        img = gaussian_filter(img, 3) #change
        img = (img > 0.001).astype(np.int32)
        # print("points: ", np.sum(img))
        # cv2.imshow("test", img*255)
        # print(np.sum(img),"cmap")
        # cv2.waitKey(0)
        # couldn't use imshow on classification maps for some reason, np.sum output is very strange,
        # not really helpful, this part needs verification
        assert img.shape == shape
        assert np.amin(img) == 0
        assert np.amax(img) == 1
        return img

    def downsample(self, img):
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4),
                         interpolation=cv2.INTER_NEAREST)
        return img

    def main(self):
        global offsets,box_sizes
        img_annots = {}
        ctr=0
        with open(self.bbx_txt, 'r+') as gt_file:
            name = gt_file.readline().strip()
            if self.img_dir_struct == "flat":
                try:
                    name = name.split("/")[1]
                except Exception:
                    print("###################################################################################################################")
            while name:
                # print(f"processing image {name}....")
                img, scale = self.read_img(name)
                img_annots[name] = []
                count = int(gt_file.readline().strip())
                while count:
                    annots = gt_file.readline().strip().split()
                    # annots = list(
                    #     map(lambda x: int(np.floor(int(x)*scale)), annots))
                    # annots = annots[:4]
                    # # remove faces with pixel boxes smaller than 16 pixels in area
                    # if annots[2]*annots[3] < 16:
                    #     count -= 1
                    #     continue
                    # center = [int(annots[0]+annots[2]/2),  # round(x1+w/2)
                    #           int(annots[1]+annots[3]/2)]  # round(y1+h/2)
                    # annots.extend(center)
                    # # so now annots are [x1, y1, box_size_x, box_size_y, center_x,center_y]
                    # img_annots[name].append(annots)
                    count -= 1
                ctr+=1
                if ctr%50==0:
                    print(ctr," done")
                name = gt_file.readline().strip()   
                if self.img_dir_struct == "flat":
                    try:
                        name = name.split("/")[1]
                    except Exception:
                        print("####################################################################################################################")             
        #         heatmap = self.generate_heatmaps(img, img_annots[name])
        #         classification_map = self.generate_cmap(heatmap)

        #         downsampled = self.downsample(classification_map)
        #         np.save(os.path.join(self.heatmap_root, name[: -4]),
        #                 downsampled)
        #         # print(f"heatmap shape: {downsampled.shape}")
        #         # assert downsampled.shape[0] >= 200 and downsampled.shape[1] >= 200
        #         offsets,box_sizes = self.generate_annotations(
        #             name[:-4], downsampled.shape, img_annots[name])
                
                
        #         def bfs(face, downsampled,offsets_bool=False):
        #                 global offsets,box_sizes
        #                 # print(offsets.dtype) 
        #                 n = [(face[0]+1, face[1]) if face[0]+1 < offsets.shape[1] else None,(face[0]-1, face[1]) if face[0]-1>=0 else None,
        #                      (face[0], face[1]+1) if face[1]+1 < offsets.shape[2] else None, (face[0], face[1]-1) if face[1]-1 >=0 else None]
        #                 for i in n:
        #                     if i is not None:
        #                         if(offsets_bool):
        #                             if offsets[0, i[0], i[1]]>0 or offsets[1, i[0], i[1]]>0:
        #                                 # print("offset edits return")

        #                                 return
        #                         else:
        #                             # print("box edits return")
        #                             if box_sizes[0, i[0], i[1]]>0 or box_sizes[1, i[0], i[1]]>0:
        #                                 return

        #                         if downsampled[i[0], i[1]] == 1:
        #                             if(offsets_bool):
        #                                 offsets[0, i[0], i[1]] = offsets[0, face[0], face[1]]
        #                                 offsets[1, i[0], i[1]] = offsets[1, face[0], face[1]]
        #                                 # print("offset edits")

        #                             else:
        #                                 # print("box edits")

        #                                 box_sizes[0, i[0], i[1]] = box_sizes[0, face[0], face[1]]
        #                                 box_sizes[1, i[0], i[1]] = box_sizes[1, face[0], face[1]]


        #                             bfs(i, downsampled,offsets_bool)

        #         # x, y = np.where(offsets[0,:, :]>0) or np.where(offsets[1, :, :]>0)
                
        #         # print(np.sum(offsets[0]))
        #         x,y = np.where(np.bitwise_or(offsets[0, :, :]>0,offsets[1, :, :]>0))
        #         faces = zip(x, y)

        #         for face in faces:
        #             bfs(face, downsampled,True)
        #         x,y = np.where(np.bitwise_or(box_sizes[0, :, :]>0,box_sizes[1, :, :]>0))
        #         faces = zip(x, y) 
        #         for face in faces:
        #             bfs(face, downsampled,False) 

        #         # print(np.sum(offsets[0]))

        #         np.save(os.path.join(self.sizes_root, name[:-4]), box_sizes)
        #         np.save(os.path.join(self.offset_root, name[:-4]), offsets)                  
        #         ctr+=1
        #         if self.img_dir_struct == "flat":
        #             try:
        #                 name = name.split("/")[1]
        #             except Exception:
        #                 print("####################################################################################################################")
        #         if ctr%50==0:
        #             print(f"{ctr}done")        
        # print(ctr)

if __name__ == "__main__":
    datagen = DataGen(img_dir_struct="flat")
    datagen.main()

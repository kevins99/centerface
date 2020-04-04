from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

# All config params during training
__C.TRAIN = edict()
__C.PREPROC = edict()
# size to which ground truth and image must be resized during training
__C.TRAIN.gt_size = ()
__C.TRAIN.inp_size = ()

# Gaussian Blur cfg
__C.PREPROC.ksize = 3
__C.PREPROC.sigmaX = 2
__C.PREPROC.sigmaY = 2

# paths
__C.PATH = edict()

# paths for dataset images
__C.PATH.ROOT = "widerface/train/"
__C.PATH.TRAIN = "data/WIDER_train/images"
__C.PATH.VAL = "data/WIDER_val/WIDER_val/images/"

# path of text file wider_face_train_bbx_gt.txt
__C.PATH.ANNOT = "wider_face_split/wider_face_train_bbx_gt.txt"

# Paths to dave datagen output


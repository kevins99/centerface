import numpy as np 
import torch
import cv2
import random


def crop(image, boxes, labels, lms, img_dim):
	"""
		Function to crop the images randomly for
		data augmentation. 
		Args: 
			image: (tensor) The accepted image
			boxes: (np array) size will be 2*(h/4)*(w/4)
			labels: 

	"""
	height, width, _ = img.shape
	pad_image_flag = True

	for _ in range(250):
		SCALES = [0.75, 0.8, 0.85, 0.9, 0.95, 1]
		scale = random.choice(SCALES)

		short_side = min(height, width)
		w = int(short_side*scale)
		h = w

		left = int(random.randrange(width-w))
		top = int(random.randrange(height-h))

		box_dim = [left, top, left+w, height+h]

		res_im = image[box_dim[0]:box_dim[2], box_dim[1]:box_dim[3]]

		
		






	
import torch
from loss import FocalLoss



criterion = FocalLoss(2)
while(1):
	x = torch.FloatTensor(10,1,8,8).uniform_(0,1)
	target = torch.randint(0,1,(10,1,8,8),dtype=torch.float32)
	# print(target)
	loss = criterion(x,target)
	if(loss<0):
		break
		print("******************************************************************************************************************************************")

	print(loss)



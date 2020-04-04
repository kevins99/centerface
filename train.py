import sys
import os
import shutil
import argparse
import json
import time
import random
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib import cm as CM

import warnings
from loss import FocalLoss,centerloss
from cf import centerface
from model_67 import SFAnet
from utils import save_checkpoint, weights_normal_init
import dataloader as dataset

parser = argparse.ArgumentParser(description='PyTorch centerface')
parser.add_argument('--train_json', default="./part_A_train.json", metavar='TRAIN',
                    help='path to train json')
parser.add_argument('--test_json', default="./part_A_test.json", metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('--gpu',default='2', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('--task',default="cf", metavar='TASK', type=str,
                    help='task id to use.')

#create log for ssh check
localtime = time.strftime("%Y-%m-%d", time.localtime())
outputfile = open("./"+localtime+".txt", 'w')
try:
    from termcolor import cprint
except ImportError:
    cprint = None
    
def log_print(text, color=None, on_color=None, attrs=None, outputfile=outputfile):
    print(text, file=outputfile)
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)
#create tensorboard dir
logDir = './tblogs'
if os.path.exists(logDir):
    shutil.rmtree(logDir)
writer = SummaryWriter(logDir)

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 12
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 4500
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 1
    args.seed = time.time()
    args.print_freq = 20
    args.save_path = './best_models'
    
    # with open(args.train_json, 'r') as outfile:        
    #     train_list = json.load(outfile)
    # with open(args.test_json, 'r') as outfile:       
    #     val_list = json.load(outfile)
    	

    # import random
    # random.seed(99)
    train_list=glob.glob("./widerface/train/images/*.jpg")
    # random.shuffle(train_list)
    # print(len(whole_list))
    # train_list=whole_list[0:1500]
    # print(whole_list[:5])
    val_list=natsorted(glob.glob("./widerface/train/images/*.jpg"))

    random.shuffle(val_list)

    # train_list=glob.glob("./UCF_CC_50/UCF_CCD-master/images/*.jpg")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = centerface()
    print(model)
    
    model = model.cuda()
    
    criterion = [nn.L1Loss(reduction='sum').cuda(),nn.BCELoss(reduction='sum').cuda(),nn.MSELoss(reduction='sum')]#centerloss()FocalLoss

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            # checkpoint['epoch']=0
            args.start_epoch = checkpoint['epoch']
            # checkpoint['best_prec1']=1e3
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            log_text = "=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch'])
            log_print(log_text, color='white', attrs=['bold'])

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,weight_decay=5e-3)

        # else:
            # weights_normal_init(model)
            # log_text = "=> no checkpoint found at '{}', use default init instead".format(args.pre)
            # log_print(log_text, color='white', attrs=['bold'])
            
    for epoch in range(args.start_epoch, args.epochs):
        
        #adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch)
        # prec1 = validate(val_list, model, criterion, epoch)
        # print(type(prec1),prec1,type(best_prec1),best_prec1)
        # is_best = prec1 < best_prec1
        # best_prec1 = min(prec1, best_prec1)
        # log_text = ' * best MAE {mae:.3f} '.format(mae=best_prec1)
        # log_print(log_text, color='red', attrs=['bold'])
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': 0,
            'optimizer' : optimizer.state_dict(),
        }, True,args.task,args.save_path)
        
    outputfile.close()


def train(train_list, model, criterion, optimizer, epoch):
    # global ssim_loss
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    size_loss_avg=AverageMeter()
    # transforms.Compose([                       transforms.Normalize(mean=[0.485, 0.456, 0.406],           std=[0.229, 0.224, 0.225]),
                   # ])
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
        ,batch_size=8,num_workers=8)
    log_text = 'epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr)
    log_print(log_text, color='green', attrs=['bold'])
    
    model.train()
    end = time.time()
    mae=0
    
    for i, (img, heatmap,offset,size)in enumerate(train_loader): #, offset, size
        
   
        data_time.update(time.time() - end)
        # print(np.array(img[0]).astype(np.uint8).shape)
        # cv2.imshow("",np.transpose(np.array(img[0]).astype(np.uint8),(1,2,0)))
        # cv2.waitKey(0)
        # print(img.shape,heatmap.shape)
        img = img.cuda()
#        print(img.shape,heatmap.shape,offset.shape,size.shape)
        heatmap=np.clip(np.round(heatmap),1e-9,0.9999999)
        cv2.imshow('asd',np.transpose(heatmap[0].numpy(),(1,2,0)))
        heatmap=heatmap.cuda().float()
        offset=offset.cuda().float()
        temp_idx=np.unravel_index(np.argmax(size[0,0]),size[0,0].shape)
        size_print=size[0,0][temp_idx[0],temp_idx[1]]
        size=size.cuda().float()
        output = model(img)
        # print(size.shape,output[1].shape)
        temp_idx=np.unravel_index(np.argmax(output[1].cpu().detach()[0,0]),size[0,0].shape)
        print(size_print,output[1].cpu().detach()[0,0][temp_idx[0],temp_idx[1]])

#        print(output[0].shape,output[1].shape,output[2].shape)     

        optimizer.zero_grad()
 
        FocalLoss=criterion[1](output[0],heatmap)

        # print(output[1].shape,output[2].shape,size.shape,offset.shape)
        # print((heatmap.squeeze(1).sum(-1).sum(-1).sum(-1)))
        size_loss=criterion[0](output[1], size)
        # print(size_loss)
        offset_loss= criterion[0](output[2], offset)
        # print(FocalLoss,size_loss,FocalLoss/size_loss)
        new_loss=(0.00001)*(size_loss)#+(1)*(offset_loss)(1)*(FocalLoss)+  use the followingk (0.01)*(FocalLoss)+
        # mae+=new_loss.item()
        losses.update(float(new_loss), 1)
        size_loss_avg.update(float(size_loss),img.size(0))
        new_loss.backward()
        optimizer.step()    

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_text = (('Epoch: [{0}][{1}/{2}]\t'
                    'Patch {patch_num:d}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.8f} ({loss.avg:.4f})\t'
                    'size_loss {size_loss.val:15f} ({size_loss.avg:3f})')
                    .format(
                    epoch, i, len(train_loader), patch_num=0, batch_time=batch_time,
                    data_time=data_time, loss=losses,size_loss=size_loss_avg))
            log_print(log_text, color='green', attrs=['bold'])
           
            
    # mae/=len(train_loader)
    # f=open("log_mse.csv","a")
    # f.write(str(epoch)+","+str(mae)+"\n")
    # writer.add_scalar('Loss/train',mae, epoch)

def validate(val_list, model, criterion, epoch):
    print('begin val')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                                      transform=transforms.Compose([
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  max_down=32,
                    output_down=2,train=False,full_img=True),num_workers=args.workers,
    batch_size=1)    
    
    model.eval()
    
    mae = 0
    mse = 0
    
    for i, (img, target,_) in enumerate(test_loader):
        # print(img.shape,target.shape)
        # patch_num = len(img_list) 
        # print(len(img_list),len(target_att_list))
        # for j, patch in enumerate(img_list):
            
        # img = img_list[j]
        # target = target_list[j]

        # target_att=target_att_list[j]

        # target_att = target_att.type(torch.FloatTensor).unsqueeze(0).cuda()

        # target_att = Variable(target_att)

        
        img = img.cuda()
        # img = Variable(img)
        with torch.no_grad():
            output = model(img)

        gt_count = target.sum(dim=(1,2,3)).type(torch.FloatTensor).cuda()
        et_count = output[0].data.sum(dim=(1,2,3))#[if att add [index]]
        # print(gt_count,et_count)
        mae += torch.sum(abs(gt_count-et_count))
        # mae += abs(output[0].detach().data.sum()-target.data.sum())#edit for batch
        
    # mae = mae/(len(test_loader))
    mae = mae/(182)

    # mse = np.sqrt(mse/(len(test_loader)))
    
    if epoch%1==0:
        log_text = ' * MAE {mae:.3f} '.format(mae=mae) #--MSE {mse:.3f}   format(,mse=mse)
        log_print(log_text, color='yellow', attrs=['bold'])
        #tensorboard
        writer.add_scalar('Loss/val', mae, epoch)
        # writer.add_image('GT',target, epoch)
        # writer.add_image('Output',output,epoch)
        # writer.add_scalar('mse', mse, epoch)

    return mae.float()
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main() 

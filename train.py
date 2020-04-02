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
from loss import FocalLoss
from cf import centerface
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

parser.add_argument('--task',default="rev", metavar='TASK', type=str,
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
    
    criterion = [nn.SmoothL1Loss(reduction='mean').cuda(),FocalLoss().cuda()]

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
        prec1 = validate(val_list, model, criterion, epoch)
        # print(type(prec1),prec1,type(best_prec1),best_prec1)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        log_text = ' * best MAE {mae:.3f} '.format(mae=best_prec1)
        log_print(log_text, color='red', attrs=['bold'])
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task,args.save_path)
        
    outputfile.close()


# ssim_loss = pytorch_ssim.SSIM(window_size = 11)
def train(train_list, model, criterion, optimizer, epoch):
    # global ssim_loss
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=None))
    log_text = 'epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr)
    log_print(log_text, color='green', attrs=['bold'])
    
    model.train()
    end = time.time()
    mae=0
    
    for i, (img, heatmap, offset, size)in enumerate(train_loader):
        
        # for j, patch in enumerate(img_list):
        # print(heatmap.shape)
        # print(img.shape,target.shape,target_att.shape)
        data_time.update(time.time() - end)
        # print(img.shape)
        # img = img_list[j]
        # target = target_list[j]
        # target_att=target_att_list[j]

        # cv2.imshow("asd",target_att.permute(1,2,0).numpy())
        # cv2.waitKey(10000)
        img=img.squeeze(0)
        heatmap=heatmap.squeeze(0)
        offset=offset.squeeze(0)
        size=size.squeeze(0)
        img = img.cuda()
        #img = Variable(img)
        # print(f"iteration: {i} shape: {img.shape}")
        output = model(img)
        heatmap=heatmap.cuda().float()
        offset=offset.cuda().float()
        size=size.cuda().float()
        # int1=int1.cuda().float()
        # int2=int2.cuda().float()
        # d_int1=d_int1.cuda().float()
        # d_int2=d_int2.cuda().float()

        # target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        # target = Variable(target)


        # target_att = target_att.type(torch.FloatTensor).unsqueeze(0).cuda()
        # target_att = Variable(target_att)

        # print(img.shape,output[0].shape,output[1].shape,target.shape,target_att.shape)
     #    plt.imshow(x.squeeze().detach().cpu().numpy(),cmap=CM.jet)
        # plt.pause(0.0001)


        
        # BCE_loss = nn.BCEWithLogitsLoss(reduction='mean')


        optimizer.zero_grad()
        # print(target.shape,output.shape,"asdasd")
        # print(type(output.item()),type(target.item()))
        # print(target_att_list[j].shape)
        # bce=BCE_loss(output[1],target_att)
        # bce=BCE_loss(output[1],target_att.clamp(0,1))
        # print(output[1].shape,target_att.shape)
        FocalLoss=criterion[1](output[0],heatmap)

        # stable_bce_loss=criterion[1](output[1],target_att)
        # if(stable_bce_loss<0):
        #     print(stable_bce_loss)

        size_loss= criterion[0](output[1], size)
        offset_loss= criterion[0](output[2], offset)


        # loss_f = criterion[0](output[2], target)

        # print((10**6)*loss,bce)
        # loss_dint1= criterion[0](output[4], d_int1)
        # loss_dint2= criterion[0](output[5], d_int2)


        # loss_int1 = criterion[1](output[2], int1)
        # loss_int2 = criterion[1](output[3], int2)

        # count_loss=abs(output[0].detach().cpu().sum().numpy()-np.sum(target))/np.sum(target)

        # gt_count = target.sum(dim=(1,2,3)).type(torch.FloatTensor).cuda()

        # et_count = output[0].data.sum(dim=(1,2,3))

        # count_loss = torch.sum(abs(gt_count-et_count))/torch.sum(et_count)

        # plt.subplot(2,1,1)
        # plt.imshow(target_att.squeeze().detach().cpu().numpy(),cmap=CM.jet)
        # plt.pause(0.0001)
        # plt.subplot(2,1,2)
        # print(loss.item())
        # plt.imshow(output[1].squeeze().detach().cpu().numpy(),cmap=CM.jet)
        # plt.pause(0.0001)



        # print(bce,loss.item())
        # print(loss)
        # ssim_loss = pytorch_ssim.SSIM(window_size = 11)
        # sim_loss=ssim_loss(output[0], target)
        # sim_loss=ssim_loss(output, target)

        # print(sim_loss.item())
# +0.001*sim_loss
        # print(loss,bce)
        # new_loss=10000*(loss+loss_dint1+loss_dint2)+100*(loss_int1+loss_int2+bce)#+0.001*(1-sim_loss)## #0*loss.item()+0.000*(1-sim_loss)+

        new_loss=(1)*(FocalLoss)+(0.1)*(size_loss+offset_loss)#+0.001*(1-sim_loss)## #0*loss.item()+0.000*(1-sim_loss)+
        mae+=new_loss.item()
        losses.update(float(new_loss), img.size(0))
        new_loss.backward()
        optimizer.step()    

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_text = (('Epoch: [{0}][{1}/{2}]\t'
                    'Patch {patch_num:d}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.8f} ({loss.avg:.4f})\t')
                    .format(
                    epoch, i, len(train_loader), patch_num=0, batch_time=batch_time,
                    data_time=data_time, loss=losses))
            log_print(log_text, color='green', attrs=['bold'])
            # writer.add_scalar('train_loss', losses.avg, epoch)
            
    mae/=len(train_loader)
    f=open("log_mse.csv","a")
    f.write(str(epoch)+","+str(mae)+"\n")
    writer.add_scalar('Loss/train',mae, epoch)

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
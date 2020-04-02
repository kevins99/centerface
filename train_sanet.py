import sys
import os
import shutil

import warnings

from model import SANet

from utilss import save_checkpoint, weights_normal_init

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np  
import argparse
import json
import cv2
import dataset
import time
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch SANet')
parser.add_argument('--train_json', default="./part_A_train.json", metavar='TRAIN',
                    help='path to train json')
parser.add_argument('--test_json', default="./part_A_test.json", metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('--gpu',default='2', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('--task',default='0', metavar='TASK', type=str,
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
logDir = './tblogs/0104'
if os.path.exists(logDir):
    shutil.rmtree(logDir)
writer = SummaryWriter(logDir)


def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    args.save_path = './best_models'
    
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = SANet()
    print(model)
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log_text = "=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch'])
            log_print(log_text, color='white', attrs=['bold'])
        else:
            weights_normal_init(model)
            log_text = "=> no checkpoint found at '{}', use default init instead".format(args.pre)
            log_print(log_text, color='white', attrs=['bold'])
            
    for epoch in range(args.start_epoch, args.epochs):
        
        #adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion, epoch)
        
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

def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       crop=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    log_text = 'epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr)
    log_print(log_text, color='green', attrs=['bold'])
    
    model.train()
    end = time.time()
    #BATCH OF HEMP< OFFSET< SCALE
    #img, thm_tgt, pff_tgt, size_tgt
    for i, (img_list, target_list)in enumerate(train_loader):
        
        
        for j, patch in enumerate(img_list):
            
            data_time.update(time.time() - end)
            
            img = img_list[j]
            target = target_list[j]
            
            img = img.cuda()
            img = Variable(img)
            output = model(img)




            target = target.type(torch.FloatTensor).unsqueeze(0).cuda() #TARGET to cuda

            target = Variable(target)

            loss = criterion(output, target) #list of criterion [focal loss, mse]
            # focal_loss = vri[0](cri[0])
            #loss = 

            losses.update(float(loss.item()), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log_text = (('Epoch: [{0}][{1}/{2}]\t'
                        'Patch {patch_num:d}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                        .format(
                        epoch, i, len(train_loader), patch_num=j, batch_time=batch_time,
                        data_time=data_time, loss=losses))
                log_print(log_text, color='green', attrs=['bold'])
                writer.add_scalar('train_loss', losses.avg, epoch)
            
    
def validate(val_list, model, criterion, epoch):
    print('begin val')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   crop=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    mse = 0
    
    for i, (img_list, target_list) in enumerate(test_loader):
        
        patch_num = len(img_list) 
        
        for j, patch in enumerate(img_list):
            
            img = img_list[j]
            target = target_list[j]
            
            img = img.cuda()
            img = Variable(img)
            with torch.no_grad():
                output = model(img)
            gt_count = target.sum().type(torch.FloatTensor).cuda()
            et_count = output.data.sum()
        
            mae += abs(gt_count-et_count)
            mse += ((gt_count-et_count)*(gt_count-et_count))
        
    mae = mae/(len(test_loader)*patch_num)
    mse = np.sqrt(mse/(len(test_loader)*patch_num))
    
    if epoch%2==0:
        log_text = ' * MAE {mae:.3f}--MSE {mse:.3f} '.format(mae=mae,mse=mse)
        log_print(log_text, color='yellow', attrs=['bold'])
        #tensorboard
        writer.add_scalar('mae', mae, epoch)
        writer.add_scalar('mse', mse, epoch)

    return mae    
        
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
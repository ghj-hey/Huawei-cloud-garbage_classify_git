import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from apex import amp

import torchcontrib
from torch.optim import lr_scheduler
#
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import Config

from utils import rubbishDataset,cutmix_data
from mymodel import BaseModel
from my_loss import LabelSmoothingLoss


def schedule(epoch):
    swa=True
    swa_start=20
    swa_lr=1e-4
    lr_init=1e-3
    t = (epoch) / (swa_start)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor

#
def train_model(model,criterion, optimizer):

    train_dataset = rubbishDataset(opt.train_val_data, opt.train_list, phase='train', input_size=opt.input_size)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=False)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    total_iters=len(trainloader)
    model_name=opt.backbone
    train_loss = []
    since = time.time()
    best_score = 0.0
    best_epoch = 0
    # accumulation_steps=4
    #
    for epoch in range(1,opt.max_epoch+1):

        lr = schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(lr)

        model.train(True)
        begin_time=time.time()
        running_corrects_linear = 0
        count=0
        for i, data in enumerate(trainloader):
            count+=1
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()

            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, 0.5, use_cuda=True)
            # print(epoch)

            #
            out_linear= model(inputs)
            _, linear_preds = torch.max(out_linear.data, 1)

            loss = criterion(out_linear, targets_a) * lam + criterion(out_linear, targets_b) * (1. - lam)
            # loss = criterion(out_linear, labels)
            #
            optimizer.zero_grad()
            # loss = loss/accumulation_steps
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()

            # if((i+1)%accumulation_steps)==0:
            optimizer.step()
            # optimizer.zero_grad()



            # if epoch>21:
                # print(optimizer.param_groups[-1]['lr'])
            if i % opt.print_interval == 0 or out_linear.size()[0] < opt.train_batch_size:
                spend_time = time.time() - begin_time
                print(
                    ' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
                train_loss.append(loss.item())
            running_corrects_linear += torch.sum(linear_preds == labels.data)
            #




        if  (epoch + 1) > 3:
            # print('swa')
            optimizer.swap_swa_sgd()
            optimizer.bn_update(trainloader, model, device='cuda')
            weight_score,val_loss = val_model(model, criterion)
        else:
            weight_score,val_loss = val_model(model, criterion)

        # scheduler.step(val_loss)

        epoch_acc_linear = running_corrects_linear.double() / total_iters / opt.train_batch_size
        print('Epoch:[{}/{}] train_acc={:.3f} '.format(epoch, opt.max_epoch,
                                                       epoch_acc_linear))


        with open(os.path.join(model_save_dir, 'log.txt'), 'a+')as f:
            f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, val_loss, weight_score))


        #
        model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + str(epoch) +'_'+str(weight_score)[:6]+ '.pth'

        #save the best model
        if weight_score > best_score:
            best_score = weight_score
            best_epoch=epoch
            best_model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + 'best'+'_'+str(weight_score)[:6]+ '.pth'
            torch.save(model.state_dict(), best_model_out_path)
            print("best epoch: {} best acc: {}".format(best_epoch,weight_score))
        #save based on epoch interval
        if epoch % opt.save_interval == 0 and epoch>opt.min_save_epoch:
            torch.save(model.state_dict(), model_out_path)

        if (epoch + 1) > 3:
            optimizer.swap_swa_sgd()

    #
    print('Best acc: {:.3f} Best epoch:{}'.format(best_score,best_epoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

@torch.no_grad()
def val_model(model, criterion):
    val_dataset = rubbishDataset(opt.train_val_data, opt.val_list, phase='val', input_size=opt.input_size)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.val_batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers)
    dset_sizes=len(val_dataset)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list=[]
    labels_list=[]
    for data in val_loader:
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=preds.cpu().numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    #
    val_acc = accuracy_score(labels_list, pres_list)
    val_loss=running_loss / dset_sizes
    print('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes,
                                                                val_acc))
    return val_acc,val_loss

if __name__ == "__main__":
    #
    opt = Config()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    criterion=LabelSmoothingLoss(classes=43,smoothing=0.1)
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    model_name=opt.backbone
    model_save_dir =os.path.join(opt.checkpoints_dir , model_name)
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    model = create_model(
        model_name,
        pretrained=True,
        num_classes=43,
        drop_rate=None,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        checkpoint_path=None)
    model.to(device)
    # model = nn.DataParallel(model)
    # model_path='./ckpt/tf_efficientnet_b5_ns_best/tf_efficientnet_b5_ns_456_best_val97.24_test96.5.pth'
    #
    # net_weight = torch.load(model_path,map_location=torch.device('cpu'))
    # model.load_state_dict(net_weight)
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # for param in model.classifier.parameters():
    #     param.requires_grad = True


    base_lr=opt.lr
    lr_ratio=10
    # ignored_params = list(map(id, model.classifier.parameters()))

    # print('the num of new layers:', len(ignored_params), flush=True)
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    # optimizer = optim.SGD([{'params': base_params},
    #                        {'params': model.classifier.parameters(), 'lr': lr_ratio*base_lr}
    #                        ], lr = base_lr, momentum=0.9)
    base_optimizer = optim.SGD((model.parameters()), lr=opt.lr, momentum=opt.MOMENTUM, weight_decay=0.0004)

    optimizer = torchcontrib.optim.SWA(base_optimizer,0,3,1e-4)

    model,optimizer=amp.initialize(model,optimizer,opt_level='O1')
    train_model(model, criterion, optimizer)




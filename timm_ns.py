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
# try:
#     from apex import amp
#     from apex.parallel import DistributedDataParallel as ApexDDP
#     from apex.parallel import convert_syncbn_model
#     has_apex = True
# except ImportError:
#     has_apex = False

# has_native_amp = False
# try:
#     if getattr(torch.cuda.amp, 'autocast') is not None:
#         has_native_amp = True
# except AttributeError:
#     pass

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
# from models import resnet50
from utils import rubbishDataset,cutmix_data,w_rubbishDataset
from mymodel import BaseModel
#
from my_loss import LabelSmoothingLoss,LabelSmoothSoftmaxCE
import numpy as np

def train_model(model,criterion, optimizer):

    train_dataset = rubbishDataset(opt.train_val_data, opt.train_list, phase='train', input_size=opt.input_size)
    # train_dataset = w_rubbishDataset(opt.train_val_data, opt.train_list, phase='train', input_size=opt.input_size)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)
    # scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_mult=2,T_0=3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    total_iters=len(trainloader)
    model_name=opt.backbone
    train_loss = []
    since = time.time()
    best_score = 0.0
    best_epoch = 0
    #
    for epoch in range(1,opt.max_epoch+1):
        model.train(True)
        begin_time=time.time()
        running_corrects_linear = 0
        count=0
        for i, data in enumerate(trainloader):
            count+=1
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()

            if np.random.rand(1)<opt.cut_prob:

                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, 1.0, use_cuda=True)
            # print(epoch)

                #
                out_linear= model(inputs)
                _, linear_preds = torch.max(out_linear.data, 1)

                loss = criterion(out_linear, targets_a) * lam + criterion(out_linear, targets_b) * (1. - lam)
            else:
                out_linear = model(inputs)
                _, linear_preds = torch.max(out_linear.data, 1)
                loss = criterion(out_linear, labels)

            # loss = criterion(out_linear, labels)
            #
            optimizer.zero_grad()
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()

            optimizer.step()

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
        weight_score,val_loss = val_model(model, criterion)
        scheduler.step()
        # scheduler.step(val_loss)

        epoch_acc_linear = running_corrects_linear.double() / total_iters / opt.train_batch_size
        print('Epoch:[{}/{}] train_acc={:.4f} '.format(epoch, opt.max_epoch,
                                                       epoch_acc_linear))
        # with open()
        with open(os.path.join(model_save_dir, 'log.txt'), 'a+')as f:
            f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, val_loss, weight_score))
        #
        model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + str(epoch) +'_'+str(weight_score)[:6]+ '.pth'
        best_model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + 'best' + '{:.4f}'.format(weight_score)+ '.pth'
        #save the best model
        if weight_score > best_score:
            best_score = weight_score
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            print("best epoch: {} best acc: {}".format(best_epoch,weight_score))
        #save based on epoch interval
        if epoch % opt.save_interval == 0 and epoch>opt.min_save_epoch:
            torch.save(model.state_dict(), model_out_path)

    #
    print('Best acc: {:.3f} Best epoch:{}'.format(best_score,best_epoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

@torch.no_grad()
def val_model(model, criterion):
    val_dataset = rubbishDataset(opt.train_val_data, opt.val_list, phase='val', input_size=opt.input_size)
    # val_dataset = w_rubbishDataset(opt.train_val_data, opt.val_list, phase='val', input_size=opt.input_size)
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
    model_path = './ckpt/tf_efficientnet_b5_ns/tf_efficientnet_b5_ns_best0.9719.pth'
    # model_path = './ckpt/tf_efficientnet_b5_ns/tf_efficientnet_b5_ns_best0.9741.pth'
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion = LabelSmoothingLoss(classes=43,smoothing=0.1)
    # criterion = LabelSmoothSoftmaxCE(lb_pos=0.9,lb_neg=0.005)
    # criterion = L.JointLoss(first=nn.crossentropyloss(), second=LabelSmoothSoftmaxCE(),
    #                         first_weight=0.5, second_weight=0.5)
    model_name=opt.backbone
    model_save_dir =os.path.join(opt.checkpoints_dir , model_name)
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    model = create_model(
        opt.backbone,
        pretrained=True,
        num_classes=43,
        drop_rate=0.1,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        checkpoint_path=None)
    model.to(device)
    net_weight = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(net_weight)
    # model = nn.DataParallel(model)


    lr_ratio = 10
    ignored_params1 = list(map(id, model.classifier.parameters()))
    ignored_params = ignored_params1

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    for param in base_params:
        param.requires_grad = False

    # optimizer = optim.SGD([{'params': base_params},
    #                        {'params': model.classifier.parameters(), 'lr': lr_ratio * opt.lr},
    #                        ], lr=opt.lr, momentum=0.9)
    # optimizer = optim.SGD((model.classifier.parameters()), lr=opt.lr, momentum=opt.MOMENTUM, weight_decay=0.0004)
    optimizer = optim.SGD((model.parameters()), lr=opt.lr, momentum=opt.MOMENTUM, weight_decay=0.0004)
    model,optimizer=amp.initialize(model,optimizer,opt_level='O1')
    train_model(model, criterion, optimizer)




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
from utils import rubbishDataset,cutmix_data
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from timm.models import create_model


def validate(val_loader, model, criterion, epoch):
    # confusion_matrix init
    cm = np.zeros((43, 43))
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(images)
            # loss = criterion(output, target)

            # compute_confusion _ added by YJ on 20191209
            cm += compute_confusion(output, target)

        confusion_file_name = os.path.join(model_save_dir, 'epoch_{}.npy'.format(epoch))
        np.save(confusion_file_name, cm)

def compute_confusion(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze()
    pred_class = torch.Tensor([int(idx_to_class[x]) for x in pred.cpu().numpy().tolist()]).int()
    target_class = torch.Tensor([int(idx_to_class[x]) for x in target.cpu().numpy().tolist()]).int()
    cm = confusion_matrix(target_class, pred_class, labels=range(43))
    return cm
if __name__ == '__main__':
    idx_to_class=[i for i in range(43)]
    opt = Config()

    val_dataset = rubbishDataset(opt.train_val_data, opt.val_list, phase='val', input_size=opt.input_size)
    val_loader = DataLoader(val_dataset,
                        batch_size=opt.val_batch_size,
                        shuffle=False,
                        num_workers=opt.num_workers)
    device = torch.device('cuda')
    model = create_model(
        'tf_efficientnet_b5_ns',
        pretrained=True,
        num_classes=43,
        drop_rate=0,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        checkpoint_path=None)
    model.to(device)
    model_path='./ckpt/tf_efficientnet_b5_ns/tf_efficientnet_b5_ns_best0.9719.pth'
    net_weight = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(net_weight)
    model_name=opt.backbone
    model_save_dir =os.path.join(opt.checkpoints_dir , model_name)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    epoch=0
    validate(val_loader,model,criterion,epoch)


    confusion_file_name = os.path.join(model_save_dir, 'epoch_{}.npy'.format(epoch))
    cm = np.load(confusion_file_name)



    import pandas as pd

    label_id_name_dict = \
        {
            "0": "其他垃圾/一次性快餐盒",
            "1": "其他垃圾/污损塑料",
            "2": "其他垃圾/烟蒂",
            "3": "其他垃圾/牙签",
            "4": "其他垃圾/破碎花盆及碟碗",
            "5": "其他垃圾/竹筷",
            "6": "厨余垃圾/剩饭剩菜",
            "7": "厨余垃圾/大骨头",
            "8": "厨余垃圾/水果果皮",
            "9": "厨余垃圾/水果果肉",
            "10": "厨余垃圾/茶叶渣",
            "11": "厨余垃圾/菜叶菜根",
            "12": "厨余垃圾/蛋壳",
            "13": "厨余垃圾/鱼骨",
            "14": "可回收物/充电宝",
            "15": "可回收物/包",
            "16": "可回收物/化妆品瓶",
            "17": "可回收物/塑料玩具",
            "18": "可回收物/塑料碗盆",
            "19": "可回收物/塑料衣架",
            "20": "可回收物/快递纸袋",
            "21": "可回收物/插头电线",
            "22": "可回收物/旧衣服",
            "23": "可回收物/易拉罐",
            "24": "可回收物/枕头",
            "25": "可回收物/毛绒玩具",
            "26": "可回收物/洗发水瓶",
            "27": "可回收物/玻璃杯",
            "28": "可回收物/皮鞋",
            "29": "可回收物/砧板",
            "30": "可回收物/纸板箱",
            "31": "可回收物/调料瓶",
            "32": "可回收物/酒瓶",
            "33": "可回收物/金属食品罐",
            "34": "可回收物/锅",
            "35": "可回收物/食用油桶",
            "36": "可回收物/饮料瓶",
            "37": "有害垃圾/干电池",
            "38": "有害垃圾/软膏",
            "39": "有害垃圾/过期药物",
            "40": "可回收物/毛巾",
            "41": "可回收物/饮料盒",
            "42": "可回收物/纸袋"
        }
    plt.figure(figsize = (30,24))
    df_cm = pd.DataFrame(cm,
                         index = [i for i in list(label_id_name_dict.keys())],
                         columns = [i for i in list(label_id_name_dict.keys())])

    sns.heatmap(df_cm, annot=True, cmap="BuPu")
    plt.savefig('confusion_b5_9719.jpg')
    plt.show()

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (30,24))
    df_cm_n = pd.DataFrame(cm_normalized,
                         index = [i for i in list(label_id_name_dict.keys())],
                         columns = [i for i in list(label_id_name_dict.keys())])

    sns.heatmap(df_cm_n, annot=True, cmap="BuPu")
    plt.savefig('confusion_b5_n_9719.jpg')
    plt.show()
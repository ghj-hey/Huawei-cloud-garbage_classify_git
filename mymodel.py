import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


__arch__ = ['resnet18', 'resnet34', 'resnet50', 'dense121', 'resnext50','resnet101','resnext50_32x4d',
            'resnext101_32x8d','resnest50','resnest101','se_resnext50_32x4d','se_resnext101_32x4d',
            'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4',
            'efficientnet-b5','efficientnet-b6','efficientnet-b7']


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super().__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)

        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def get_efficientnet(model_name='efficientnet-b0', num_classes=5000, pretrained=True):
    if pretrained:
        net = EfficientNet.from_pretrained(model_name)
    else:
        net = EfficientNet.from_name(model_name)
    net._avg_pooling = AdaptiveConcatPool2d()
    in_features = net._fc.in_features
    # net._fc = nn.Sequential(nn.Linear(in_features*2, in_features, bias=True),
    #                           nn.ReLU(),
    #                           nn.Dropout(p=0.5),
    #                           nn.Linear(in_features, num_classes, bias=True))
    net._fc = nn.Linear(in_features=in_features*2, out_features=num_classes, bias=True)

    return net
class BaseModel1(nn.Module):
    def __init__(self, model_name, num_classes=200, pretrained=True, pool_type='cat', down=True):
        super().__init__()
        assert model_name in __arch__
        self.model_name = model_name

        if model_name == 'efficientnet-b7' or model_name == 'efficientnet-b6'\
                or model_name == 'efficientnet-b5' or model_name == 'efficientnet-b4'\
                or model_name == 'efficientnet-b3' or model_name == 'efficientnet-b2'\
                or model_name == 'efficientnet-b1' or model_name == 'efficientnet-b0':
            if model_name == 'efficientnet-b0':
                backbone = nn.Sequential(
                    get_efficientnet(model_name=model_name, num_classes=num_classes))
            if model_name == 'efficientnet-b3':
                backbone = nn.Sequential(
                    get_efficientnet(model_name=model_name, num_classes=num_classes))
            if model_name == 'efficientnet-b4':
                backbone = nn.Sequential(
                    get_efficientnet(model_name=model_name, num_classes=num_classes))

            #b0 plane = 1280,b3 plane = 1536,b4 plane = 1792
        else:
            backbone = None
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        return out

class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes=200, pretrained=True, pool_type='cat', down=True):
        super().__init__()
        assert model_name in __arch__
        self.model_name = model_name

        if model_name == 'resnet50' or model_name == 'resnet18' or model_name == 'resnet152' \
                or model_name == 'resnext101_32x8d' or model_name == 'resnext50_32x4d' \
                or model_name == 'resnest50' or model_name == 'resnest101' or model_name == 'dense121':
            if model_name == 'resnet50':
                backbone = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
                plane = 2048
            if model_name == 'resnet18':
                backbone = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])
                plane = 512
            if model_name == 'resnext101_32x8d':
                backbone = nn.Sequential(*list(models.resnext101_32x8d(pretrained=pretrained).children())[:-2])
                plane = 2048
            if model_name == 'resnext50_32x4d':
                backbone = nn.Sequential(*list(models.resnext50_32x4d(pretrained=pretrained).children())[:-2])
                plane = 2048
            if model_name == 'dense121':
                backbone = nn.Sequential(*list(models.densenet121(pretrained=pretrained).features.children()))
                plane = 1024

            if model_name == 'resnest50':
                backbone = nn.Sequential(
                    *list(torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True).children())[:-2])
                plane = 2048
            if model_name == 'resnest101':
                backbone = nn.Sequential(
                    *list(torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True).children())[:-2])
                plane = 2048
        elif model_name == 'se_resnext101_32x4d' or model_name == 'se_resnext50_32x4d' :
            if model_name == 'se_resnext101_32x4d':
                backbone = nn.Sequential(
                    *list(pretrainedmodels.se_resnext101_32x4d(pretrained='imagenet').children())[:-2])
                plane = 2048
            if model_name == 'se_resnext50_32x4d':
                backbone = nn.Sequential(
                    *list(pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet').children())[:-2])
                plane = 2048
        elif model_name == 'efficientnet-b7' or model_name == 'efficientnet-b6'\
                or model_name == 'efficientnet-b5' or model_name == 'efficientnet-b4'\
                or model_name == 'efficientnet-b3' or model_name == 'efficientnet-b2'\
                or model_name == 'efficientnet-b1' or model_name == 'efficientnet-b0':
            if model_name == 'efficientnet-b0':
                backbone = EfficientNet.from_pretrained('efficientnet-b0')
                plane = 1280
            if model_name == 'efficientnet-b1':
                backbone = EfficientNet.from_pretrained('efficientnet-b1')
                plane = 1280

            if model_name == 'efficientnet-b3':
                backbone = EfficientNet.from_pretrained('efficientnet-b3')
                plane = 1536

            if model_name == 'efficientnet-b4':
                backbone = EfficientNet.from_pretrained('efficientnet-b4')
                plane = 1792

            if model_name == 'efficientnet-b5':
                backbone = EfficientNet.from_pretrained('efficientnet-b5')
                plane = 2048
            if model_name == 'efficientnet-b6':
                backbone = EfficientNet.from_pretrained('efficientnet-b6')
                plane = 2304
            if model_name == 'efficientnet-b7':
                backbone = EfficientNet.from_pretrained('efficientnet-b7')
                plane = 2560

            # self.model = EfficientNet.from_pretrained(model_name)
            # self.model = get_efficientnet(model_name=model_name, num_classes=num_classes)
            # backbone = nn.Sequential(*list(EfficientNet.from_pretrained(model_name).children())[:-2])
            # plane = 1536
            #b0 plane = 1280,b3 plane = 1536,b4 plane = 1792
        else:
            backbone = None
            plane = None

        self.backbone = backbone

        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == 'cat':
            self.pool = AdaptiveConcatPool2d()
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool_type == 'gem':
            self.pool = GeneralizedMeanPooling()
        else:
            self.pool = None

        if down:
            if pool_type == 'cat':
                self.down = nn.Sequential(
                    nn.Linear(plane * 2, plane),
                    nn.BatchNorm1d(plane),
                    nn.Dropout(0.2),
                    nn.ReLU(True)
                )
            else:
                self.down = nn.Sequential(
                    nn.Linear(plane, plane),
                    nn.BatchNorm1d(plane),
                    nn.Dropout(0.2),
                    nn.ReLU(True)
                )
        else:
            self.down = nn.Identity()

        self.se = SELayer(plane)
        self.hidden = nn.Linear(plane, plane)
        self.relu = nn.ReLU(True)

        self.metric = nn.Linear(plane, num_classes)
        # self.metric = AddMarginProduct(plane, num_classes)

    def forward(self, x):
        if self.model_name == 'efficientnet-b3' or self.model_name == 'efficientnet-b4' \
                or self.model_name == 'efficientnet-b0' or self.model_name == 'efficientnet-b1'\
                or self.model_name == 'efficientnet-b5'or self.model_name == 'efficientnet-b6' \
                or self.model_name == 'efficientnet-b7':
            feat = self.backbone.extract_features(x)
            # print(feat)
        else:
            feat = self.backbone(x)
        # feat = self.pool(feat)
        # se = self.se(feat).view(feat.size(0), -1)
        # feat_flat = feat.view(feat.size(0), -1)
        # feat_flat = self.relu(self.hidden(feat_flat) * se)
        #
        # out = self.metric(feat_flat)
        # return out

        feat_flat = self.pool(feat).view(feat.size(0), -1)
        feat_flat = self.down(feat_flat)
        out = self.metric(feat_flat)

        return out


if __name__ == '__main__':
    model = BaseModel(model_name='resnest50', num_classes=43, pretrained=True,
                      pool_type='cat', down=1)
    # model = BaseModel(model_name='resnet101').eval()
    # x = torch.randn((1, 3, 224, 224))
    # out = model(x)
    # print(out.size())
    print(model)
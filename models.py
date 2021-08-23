import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torch.nn import Parameter
from gcn import GraphConvolution
from breg_next import BReGNeXt
from utils import gen_A, gen_adj

class Emotion_GCN(nn.Module):
    """
    Based on the code of ML-GCN https://github.com/Megvii-Nanjing/ML-GCN
    """

    def __init__(self, adj_file=None, in_channel=300, input_size=227):
        super(Emotion_GCN, self).__init__()
        self.features = models.densenet121(pretrained=True).features
        if input_size == 227:
            self.pooling = nn.MaxPool2d(7, 7)
        else:
            self.pooling = nn.MaxPool2d(3, 3)

        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 1024)
        self.relu = nn.LeakyReLU(0.2)
        
        #self.gc = GraphConvolution(in_channel, 1024) # 1 layer

        #self.gc1 = GraphConvolution(in_channel, 512) # 3 layers
        #self.gc2 = GraphConvolution(512, 512)
        #self.gc3 = GraphConvolution(512, 1024)
        #self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        print(self.A)

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        #x = self.gc(inp, adj) # 1 layer
        
        #x = self.gc1(inp, adj) # 3 layers
        #x = self.relu(x)
        #x = self.gc2(x, adj)
        #x = self.relu(x)
        #x = self.gc3(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        return x[:, :7], x[:, 7:]



class multi_densenet(nn.Module):
    def __init__(self, pretrained=True, num_categorical=7):
        super(multi_densenet, self).__init__()

        self.model_base = models.densenet121(pretrained=pretrained).features
        self.num_categorical = num_categorical
        self.num_continuous = 2

        self.lin_cat = nn.Linear(1024, self.num_categorical)
        self.lin_cont = nn.Linear(1024, self.num_continuous)
    
    def forward(self, x):
        
        feat = self.model_base(x)
        out = F.relu(feat, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out_cont = self.lin_cont(out)
        out_cat = self.lin_cat(out)
        
        return out_cat, out_cont



class BReGNeXt_GCN(nn.Module):

    def __init__(self, adj_file=None, in_channel=300):
        super(BReGNeXt_GCN, self).__init__()
        self.model = BReGNeXt(n_classes=7)

        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 128)
        self.relu = nn.LeakyReLU(0.2)
        
        _adj = gen_A(adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        print(self.A)

    def forward(self, feature, inp):
        feature = torch.nn.functional.pad(feature, (1,1,1,1,0,0))
        feature = self.model._conv0(feature)
        feature = self.model._model(feature).reshape(-1, 128)

        inp = inp[0]
        adj = gen_adj(self.A).detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        return x[:, :7], x[:, 7:]

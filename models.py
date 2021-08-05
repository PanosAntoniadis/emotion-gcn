import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision.models as models

from gcn import GraphConvolution

from utils import gen_A, gen_adj

class Emotion_GCN(nn.Module):
    """
    Based on the code of ML-GCN https://github.com/Megvii-Nanjing/ML-GCN
    """

    def __init__(self, adj_file=None, in_channel=300):
        super(Emotion_GCN, self).__init__()
        self.features = models.densenet121(pretrained=True).features
        self.pooling = nn.MaxPool2d(7, 7)

        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 1024)
        self.relu = nn.LeakyReLU(0.2)

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

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)

        return x[:, :7], x[:, 7:]

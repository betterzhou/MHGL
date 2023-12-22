import torch.nn as nn
import torch.nn.functional as F
from layers_GCN import GraphConvolution


class GCN_vanilla(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_vanilla, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 20*nhid)
        self.gc2 = GraphConvolution(20*nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        emb = self.gc2(x, adj)

        return emb


class GCN_vanilla_3_layers(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_vanilla_3_layers, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 128)
        self.gc3 = GraphConvolution(128, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        emb = self.gc3(x, adj)
        return emb


class GCN_vanilla_4_layers(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_vanilla_4_layers, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 256)
        self.gc3 = GraphConvolution(256, 128)
        self.gc4 = GraphConvolution(128, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        emb = self.gc4(x, adj)

        return emb


class GCN_vanilla_5_layers(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_vanilla_5_layers, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 256)
        self.gc3 = GraphConvolution(256, 128)
        self.gc4 = GraphConvolution(128, 64)
        self.gc5 = GraphConvolution(64, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        emb = self.gc5(x, adj)

        return emb



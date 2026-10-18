import torch
from torch import nn
from torch.nn import functional as F

MORPH_COLS = ["Area", "Perimeter", "Circularity", "Aspect_Ratio", "Convex_Area",
              "Solidity", "Equivalent_Diameter", "Eccentricity", "Bbox_Area", "Extent",
              "Major_Axis_Length", "Minor_Axis_Length", "Orientation",
              "Inertia_Tensor_Eigvals_X", "Inertia_Tensor_Eigvals_Y", "Convexity",
              "Irregularity_Index"]


class FusionGate(nn.Module):
    def __init__(self, ld=96, gd=1024, out=1024):
        super().__init__()
        self.W_l = nn.Linear(ld, out)
        self.W_g = nn.Linear(gd, out)
        self.gate = nn.Linear(ld + gd, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, x_l, x_g):
        gate = torch.sigmoid(self.gate(torch.cat([x_l, x_g], -1)))
        return self.norm(self.W_l(x_l) + gate * self.W_g(x_g))


class FusionClassifier(nn.Module):
    def __init__(self, n_classes=5, local_dim=96, global_dim=1024):
        super().__init__()
        self.fusion = FusionGate(local_dim, global_dim)
        self.head = nn.Linear(1024, n_classes)

    def forward(self, x_l, x_g):
        return self.head(self.fusion(x_l, x_g))


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.4, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)
        self.a = nn.Linear(2 * out_f, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, h, adj):
        wh = self.W(h)
        batch, nodes, dim = wh.shape
        hi = wh.unsqueeze(2).expand(batch, nodes, nodes, dim)
        hj = wh.unsqueeze(1).expand(batch, nodes, nodes, dim)
        logits = self.leaky(self.a(torch.cat([hi, hj], -1)).squeeze(-1))
        logits = logits.masked_fill(adj == 0, -1e9)
        attention = F.dropout(torch.softmax(logits, -1), self.dropout, self.training)
        return F.elu(attention @ wh)


class CellularGAT(nn.Module):
    def __init__(self, in_dim=1024, n_classes=5, dropout=0.4):
        super().__init__()
        self.proj = nn.Linear(in_dim, 100)
        self.heads = nn.ModuleList([GraphAttentionLayer(100, 128, dropout) for _ in range(8)])
        self.out = GraphAttentionLayer(128 * 8, 256, dropout)
        self.classifier = nn.Linear(256, n_classes)
        self.dropout = dropout

    def embed(self, x, adj):
        h = F.elu(self.proj(x))
        h = torch.cat([head(h, adj) for head in self.heads], -1)
        h = F.dropout(h, self.dropout, self.training)
        return self.out(h, adj)

    def forward(self, x, adj):
        return self.classifier(self.embed(x, adj))


def make_mlp(in_dim, n_classes, dropout=0.3):
    model = nn.Sequential()
    previous = in_dim
    for width in (512, 256, 128, 64):
        model.append(nn.Linear(previous, width))
        model.append(nn.BatchNorm1d(width))
        model.append(nn.ReLU())
        model.append(nn.Dropout(dropout))
        previous = width
    model.append(nn.Linear(previous, n_classes))
    return model

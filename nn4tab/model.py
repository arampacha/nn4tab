# AUTOGENERATED! DO NOT EDIT! File to edit: 02_model.ipynb (unless otherwise specified).

__all__ = ['noop', 'LinearBlock', 'trunc_normal_', 'Embedding', 'TabInputBlock', 'TabularModel', 'get_tabular_model',
           'emb_sz_rule', 'emb_sizes']

# Cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gc
import typing
from typing import Sequence, Union, Tuple, List

# Cell
def noop(x):
    return x

# Cell
class LinearBlock(nn.Module):
    def __init__(self, ni:int, no:int, bn:bool=True, drop:float=0., activation:Union[str, None]='relu'):
        super().__init__()
        m = []
        if bn: m.append(nn.BatchNorm1d(ni))
        m.append(nn.Linear(ni, no))
        if drop: m.append(nn.Dropout(drop))
        if activation=='relu': m.append(nn.ReLU())
        self.m = nn.Sequential(*m)
    def forward(self, x):
        return self.m(x)

# Cell
## Modified Embedding layer like in fast.ai lib https://github.com/fastai/fastai
def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, **kwargs):
        super().__init__(ni, nf, **kwargs)
        trunc_normal_(self.weight.data, std=0.01)

# Cell
class TabInputBlock(nn.Module):

    def __init__(self, emb_sz:Sequence[Tuple], n_cont:int, emb_drop:float=0.):
        super().__init__()
        self.n_cont = n_cont
        self.bn = nn.BatchNorm1d(n_cont) if n_cont else noop
        self.embs = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_sz])
        self.emb_drop = nn.Dropout(emb_drop) if emb_drop else noop

    def forward(self, x_cat, x_cont):
        # mb rewrite without python list compr
        if self.embs:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn(x_cont)
            x = torch.cat([x, x_cont], 1) if self.embs else x_cont
        return x

# Cell
class TabularModel(nn.Module):

    def __init__(self, layers:Sequence[int], emb_sz:Sequence[Tuple], n_cont:int, n_out:int,
                 emb_drop:float=0., drops:Sequence=[]):
        super().__init__()
        self.stem = TabInputBlock(emb_sz, n_cont, emb_drop)

        layers = [sum([x[1] for x in emb_sz]) + n_cont] + layers
        if not drops: drops = [0. for _ in layers]
        lins = [LinearBlock(ni, no, drop=p) for ni, no, p in zip(layers[:-1], layers[1:], drops)]
        lins.append(LinearBlock(layers[-1], n_out, activation=None))
        self.lins = nn.Sequential(*lins)

    def forward(self, x_cat, x_cont):
        return self.lins(self.stem(x_cat, x_cont))

# Cell
def get_tabular_model(dataset, n_out, cont_names=[], cat_names=[],  layers=[200,100], emb_voc={}, emb_drop=0.1, drops=[]):
    if not (cont_names or cat_names):
        cont_names, cat_names = dataset.cont_names, dataset.cat_names
    emb_sz = emb_sizes(dataset, cat_names, emb_voc)
    return TabularModel(layers, emb_sz, len(cont_names), n_out, emb_drop=emb_drop, drops=drops)

# Cell
# used in fastai lib
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))

# Cell
def emb_sizes(dataset, cat_names, emb_voc={}):
    cat_sz = [len(dataset.data[col].unique())+1 for col in cat_names]
    emb_sz = [emb_voc.get(col, emb_sz_rule(cat_sz[i])) for i, col in enumerate(cat_names)]
    return list(zip(cat_sz, emb_sz))
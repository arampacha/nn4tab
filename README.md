# nn4tab
> A minimal implementation of neural network for tabular data.


<a href="https://github.com/arampacha/nn4tab/blob/master/_example_adult-clean.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Currently under development.

## Install

not ready yet

```
pip install nn4tab
```

instead do:

```
git clone https://github.com/arampacha/nn4tab
pip install nn4tab
```

## How to use

```python
import pandas as pd
import torch
from torch import nn
```

Load data:

```python
df = pd.read_csv(r'./datasets/adult_preproc.csv', index_col=0)
```

Specify target variable and lists of continuous and categorical variables. Later can be sugested automaticaly.

```python
dep_var = ['salary']
cont, cat = cont_cat_split(df, dep_var)
```

Get training and validation datasets and dataloaders for the data:

```python
train_ds, valid_ds = get_dsets(df, cont, cat, dep_var)
```

```python
train_dl = get_dl(train_ds)
valid_dl = get_dl(valid_ds, train=False)
```

```python
def get_dataloaders():
    return get_dl(train_ds), get_dl(valid_ds)
dataloaders = get_dataloaders()
```

Create a tabula neural network:

```python
tab_nn = get_tabular_model(train_ds, 1, layers=[100, 50])
```

And a learner which combines data, model, opimizer, loss function - everything you need to train NN.

```python
learn = LearnerV0(tab_nn, dataloaders, torch.optim.Adam, nn.BCEWithLogitsLoss(), metrics=accuracy_binary)
```

Fit the model to training data and check the performance on validataion data.

```python
learn.fit(1)
```

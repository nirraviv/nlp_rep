from fastai.text import *
from pathlib import Path

download_data = True
train_lm = True
train_sa = True
model_name = 'transformer'  # transformer | awd_lstm | transformer_xl

path = untar_data(URLs.IMDB)
output_dir = Path(r'./output')

batch_size = 32
lm_lr0 = 1e-3
lm_lr1 = 1e-4
cls_lr0 = 1e-2
cls_lr1 = 2e-3
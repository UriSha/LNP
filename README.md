# Lingual Neural Processes
This repository contains the code and data for our final project.

### Prerequisites
In order to train a model from scrach you need to have the following:
- python 3.7
- torch
- numpy
- skylearn
- pytorch-pretrained-bert

In addition, you need fasttext word embeddings file, get it [here](https://fasttext.cc/docs/en/english-vectors.html).
- Download wiki-news-300d-1M.vec.zip
- Unzip it and name the file fasttext.vec
- locate fasttext.vec in data/embeddings/

### Training the model
```shell
$ python main.py --data_file APRC.txt --mask_ratios 0.25 0.5 --learning_rate 0.0003 --test_size 0.1 --epochs 500 --to_cuda True
```
You can control and change the hyper-parameters to try different settings.

In order to run without a gpu set the cuda parameter to False

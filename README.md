# More than Text: Multi-modal Chinese Word Segmentation


This repository contains the source code and dataset for the paper: More than Text: Multi-modal Chinese Word Segmentation. Dong Zhang, Zheng Hu, Shoushan Li, Hanqian Wu, Qiaoming Zhu and Guodong Zhou. ACL 2021.

## Install
* python=3.7
* pytorch=1.7
* transformers=3.4.0
* tqdm
* boto3
* numpy 
* pytorch-crf

## Dataset
In our paper, we use the data under the `./Data/small_version` directory. The `./Data/extended_version` 
directory contains the unannotated data which will be used to perform semi-supervised or 
unsupervised learning in the future.



## Usage
We use `main.py` to train and test the model.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--bert_model`: the directory of pre-trained BERT model.
* `--model_name`: the name of model to save.
* `--audio_encoder_config_path`:  the config path of audio_encoder.
* `--num_multi_attention_layers`: the number of multi_attention_layers for multi-attention gating mechanism.





# SAMS_VQA

## Description 

This is project SAMS_VQA developed by team SAMS composed of Arda Bati, Marjan Emadi, So Sasaki, and Sina Shahsavari. The repository mainly consists of two implementations of Visual Question Answering (VQA). The experiment1 is an implementation for Bottom-Up and Top-Down Attention for VQA, which our final report is based on. The experiment2 is a completely different implementation for a vanilla VQA. The details are on the README file in each experiment directory.  

## Requirements and Usage

### Experiment1

The experiment1 requires 64G memory. To get sufficient resource on ing6 server, create a pod as follows:
```
launch-pytorch-gpu.sh -m 64
```

Use python 2.7 and install packages pillow and h5py. Since CUDA (Version 8) and pytorch (0.3.1) of DSMLP Python 2.7 pod is imcompatible, you need to downgrade pytorch to 0.3.0. 
```
conda create -n envname python=2.7 mkl=2018 pytorch=0.3.0 -c pytorch -c intel
source activate envname
pip install --user pillow h5py
```

To train the model, run the followings:
```
cd experiment1
sh tools/download.sh
sh tools/process.sh
python main.py
```

For demonstration, you need the experiment results, our_answers.dms. This file is uploaded in experiment1/demo, but you can also generate it as follows:
```
cd experiment1/demo
python demo.py
```

Then run the demo script on jupyter notebook:

- experiment1/demo/Demo.ipyenb

### Experiment2

For experiment2, use python 3.7 and install packages torchtext, tensorboardX, and utils. To execute the code, run the followings:

```
cd experiment2
pip install --user -r requirements.txt
mkdir results
mkdir preprocessed
mkdir preprocessed/img_vgg16feature_train
mkdir preprocessed/img_vgg16feature_val
python main.py -c config.yml
```

To skip preprocessing after the first execution, disable 'preprocess' in the config file.

The experiment2 does not include demo scripts or trained model parameters.


## Code organization 

### experiment1

 - experiment1: An implementation for Bottom-Up and Top-Down Attention for VQA
 - experiment1/main.py: Main script to set up models and dataloaders and run training module
 - experiment1/train.py: Module for training and evaluation
 - experiment1/base_model.py: Module for BaseModel, which controls WordEmbedding, QuestionEmbedding, Attention, FCnet, and classifier
 - experiment1/language_model.py: Module for WordEmbedding and QuestionEmbedding
 - experiment1/fc.py: Module for fully-connected network part
 - experiment1/attention.py: Module for Attention
 - experiment1/dataset.py: Module for dictionary and dataloading
 - experiment1/utils.py: Module for utility scripts
 - experiment1/demo: Files related to demo
 - experiment1/demo/Demo.ipynb: Main notebook for demo 
 - experiment1/demo/demo.py: Script to generate answers
 - experiment1/demo/base_model.py: Module for BaseModel
 - experiment1/demo/glove6b_init_300d.npy: Parameters for WordEmbedding
 - experiment1/demo/model.pth: Trained model
 - experiment1/demo/our_answers.dms: Result answers for sample questions
 - experiment1/demo/test.dms: Image features for sample questions
 - experiment1/demo/trainval_label2ans.pkl: Dictionary to decode answers
 - experiment1/demo/readme.md: Other information for demo
 - experiment1/tools: Files for preprocessing and downloading
 - experiment1/tools/download.sh: Script to download raw image features
 - experiment1/tools/process.sh: Script to run create_dictionary, compute_softscore, and detection_features_converter
 - experiment1/tools/create_dictionary.py: Script to create dictionary
 - experiment1/tools/compute_softscore.py: Script to compute softscore
 - experiment1/tools/detection_features_converter.py: Script to generate detection features
 - experiment1/data: Data files
 - experiment1/data/train_ids.pkl: Indeces of training data
 - experiment1/data/val_ids.pkl: Indeces of validation data
 - experiment1/README.md: Information for experiment1

### experiment2

 - experiment2: An implementation for a vanilla VQA.
 - experiment2/main.py: Main script to manage the training and validation of the model
 - experiment2/train.py: Module for training and validation
 - experiment2/vqa.py:  Module for the neural network architecture
 - experiment2/dataset.py: Module for data loading
 - experiment2/config.yml: Configuration file
 - experiment2/requirements.txt: List of the requirements
 - experiment2/README.md: Information for experiment2

### Miscellaneous scripts

 - misc: Miscellaneous scripts
 - misc/data_format_check.ipynb: Script for preliminary data visualization 
 - misc/some_useful_codes: Scripts which were not used after all



## References

### Experiment1

 - https://arxiv.org/abs/1707.07998 
 - https://github.com/hengyuan-hu/bottom-up-attention-vqa. 
 - http://www.visualqa.org/challenge.html
 - http://www.visualqa.org/evaluation.html
 - https://arxiv.org/pdf/1611.09978.pdf

### Experiment2

 - https://github.com/Shivanshu-Gupta/Visual-Question-Answering
 - https://vqa.cloudcv.org/
 - https://arxiv.org/abs/1505.00468
 - https://arxiv.org/pdf/1511.02274
 - https://arxiv.org/abs/1705.06676
 - http://visualqa.org/download.html
 - https://github.com/Shivanshu-Gupta/Visual-Question-Answering/config

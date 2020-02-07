# Experiment2

## Description 

The experiment2 is a vanilla VQA implementation. These scripts are originally from https://github.com/Shivanshu-Gupta/Visual-Question-Answering, but they are significantly modified. 

## Requirements and Usage

Use python 3.7 and install packages torchtext, tensorboardX, and utils. To execute the code, run the followings:

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

The experiment2 does not include demo scripts.


## Code organization 

 - experiment2: An implementation for a vanilla VQA.
 - experiment2/main.py: Main script to manage the training and validation of the model
 - experiment2/train.py: Module for training and validation
 - experiment2/vqa.py:  Module for the neural network architecture
 - experiment2/dataset.py: Module for data loading
 - experiment2/config.yml: Configuration file
 - experiment2/requirements.txt: List of the requirements
 - experiment2/README.md: Other information


## Referenced sites and papers

 - https://github.com/Shivanshu-Gupta/Visual-Question-Answering
 - https://vqa.cloudcv.org/
 - https://arxiv.org/abs/1505.00468
 - https://arxiv.org/pdf/1511.02274
 - https://arxiv.org/abs/1705.06676
 - http://visualqa.org/download.html
 - https://github.com/Shivanshu-Gupta/Visual-Question-Answering/config

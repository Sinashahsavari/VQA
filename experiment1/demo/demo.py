import sys
sys.path.append("..")

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from dataset import Dictionary, VQAFeatureDataset
import base_model
import pickle
import os.path


constructor = 'build_baseline0_newatt'
model = getattr(base_model, constructor)( 1024).cuda()
dict_pth3='./glove6b_init_300d.npy'
model.w_emb.init_embedding(dict_pth3)

model = nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('model.pth'))

test_samples= open ('test.dms','r')
tests=pickle.load(test_samples)
test_samples.close()

v=tests[0]
b=tests[1]
q=tests[2]
preds=[]

for i in range(len(v)):
    preds.append(model(v[i].cuda(),b[i].cuda(),q[i].cuda(),None))
            

file= open ('our_answers.dms','w')
pickle.dump(preds,file)
file.close()

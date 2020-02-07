import json
import string
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class VQADataset(torch.utils.data.Dataset):

    filebase = '/datasets/ee285f-public/VQA2017/'
    ques_vocab = {}
    ans_vocab = {}
    wtoi_question = {}
    wtoi_answer = {}
    def __init__(self, mode='train', preprocess=True):

        if len(VQADataset.ques_vocab) == 0:
            if preprocess:
                self._preprocess()
            with open('preprocessed/itow_question.json', 'r') as f: VQADataset.ques_vocab    = json.load(f)
            with open('preprocessed/wtoi_question.json', 'r') as f: VQADataset.wtoi_question = json.load(f)
            with open('preprocessed/itow_answer.json',   'r') as f: VQADataset.ans_vocab     = json.load(f)
            with open('preprocessed/wtoi_answer.json',   'r') as f: VQADataset.wtoi_answer   = json.load(f)

        self.mode = mode
        self.data = self._read_data(mode)
        self.data_encoded = self._data_encoder(self.data, VQADataset.wtoi_question, VQADataset.wtoi_answer)

    def __len__(self):
        return len(self.data_encoded)

    def __getitem__(self, idx): 
        data = self.data_encoded[idx]
        question    = data['question']
        answer      = data['answer']
        image_id    = data['image_id']
        question_id = data['question_id']
        img_feature_filename = 'preprocessed/img_vgg16feature_'+self.mode+'/'+str(image_id).zfill(12)+'.pt'
        img_feature = torch.load(img_feature_filename)
        return torch.from_numpy(np.array(question)), img_feature, image_id, answer, question_id

    def _read_data(self, mode="train"):
        data = []
        question_filename = 'v2_OpenEnded_mscoco_'+mode+'2014_questions.json'
        annotation_filename = "v2_mscoco_"+mode+"2014_annotations.json"
        with open(self.filebase+question_filename,'r') as f:
            data_question = json.load(f)
        with open(self.filebase+annotation_filename,'r') as f:
            data_annotation = json.load(f)
        for i in range(len(data_annotation['annotations'])):
            image_id = data_question['questions'][i]["image_id"]
            data.append({'question':data_question['questions'][i]['question'], 
                         'annotation': data_annotation['annotations'][i]['answers'][0]['answer'],
                         'image_id': image_id,
                         'question_id': data_question['questions'][i]['question_id']})
        return data
    
    def _build_vocab(self, data):
        translator = str.maketrans({key: None for key in string.punctuation}) # to remove punctuation
        vocab_question = defaultdict(int)
        vocab_answer   = defaultdict(int)
        for i in data:
            question = i['question'].translate(translator).lower().split()
            answer   = i['annotation'].lower()
            for j in question:
                vocab_question[j] += 1
            vocab_answer[answer] += 1
        vocab_question_ordered = sorted(vocab_question.items(), key=lambda i: i[1], reverse=True)
        vocab_answer_ordered   = sorted(vocab_answer.items(),   key=lambda i: i[1], reverse=True)
        itow_question = {str(i+1):w   for i,(w,f) in enumerate(vocab_question_ordered)}
        wtoi_question = {       w:i+1 for i,(w,f) in enumerate(vocab_question_ordered)}
        itow_answer   = {str(i+1):w   for i,(w,f) in enumerate(vocab_answer_ordered)}
        wtoi_answer   = {       w:i+1 for i,(w,f) in enumerate(vocab_answer_ordered)}
        itow_question[str(0)] = '<UNK>'
        wtoi_question['<UNK>'] = 0
        itow_answer[str(0)] = '<UNK>'
        wtoi_answer['<UNK>'] = 0
        return itow_question, wtoi_question, itow_answer, wtoi_answer
    
    def _data_encoder(self, data, wtoi_question, wtoi_answer):
        translator = str.maketrans({key: None for key in string.punctuation}) # to remove punctuation
        data_encoded = []
        for i in data:
            question = i['question'].translate(translator).lower().split() # + '<EOS>' if needed...
            answer   = i['annotation'].lower()
            question_encoded = [ wtoi_question[w] if w in wtoi_question.keys() else 0 for w in question ]
            data_encoded.append({'question': question_encoded,
                                 'answer': wtoi_answer[answer] if answer in wtoi_answer.keys() else 0,
                                 'image_id': i['image_id'],
                                 'question_id': i['question_id']})
        return data_encoded

    def _preprocess(self):
        print('preprocessing vocab and train/val images...')
        data_train = self._read_data("train") 
        data_val = self._read_data("val") 

        ques_vocab, wtoi_question, ans_vocab, wtoi_answer  = self._build_vocab(data_train)
        with open('preprocessed/itow_question.json', 'w') as f: json.dump(ques_vocab,    f, indent=4)
        with open('preprocessed/wtoi_question.json', 'w') as f: json.dump(wtoi_question, f, indent=4)
        with open('preprocessed/itow_answer.json',   'w') as f: json.dump(ans_vocab,     f, indent=4)
        with open('preprocessed/wtoi_answer.json',   'w') as f: json.dump(wtoi_answer,   f, indent=4)
        print('vocab preprocessed.')

        img_scale=(256, 256)
        img_crop=224
        transform = transforms.Compose([
            transforms.Resize(img_scale), #transforms.Scale(img_scale), #sasaki
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("preprocessing on device:", device)
        extractor = models.vgg16(pretrained=True)
        for param in extractor.parameters():
            param.requires_grad = False
        extactor_fc_layers = list(extractor.classifier.children())[:-1]
        #if image_channel_type.lower() == 'normi':
        #    extactor_fc_layers.append(Normalize(p=2))
        extractor.classifier = nn.Sequential(*extactor_fc_layers)
        extractor.to(device)

        for mode in ['train', 'val']:
            print('preprocessing '+mode+' image features...')
            if mode == 'train': data = data_train
            elif mode == 'val': data = data_val

            batch_size = 500
            image_id_done = set([])
            for i in range(0,len(data),batch_size):
                imgs = []
                image_ids = []
                end = min(i+batch_size, len(data))
                for j in range(i, end):
                    image_id = data[j]['image_id']
                    if image_id in image_id_done: continue
                    image_id_done.add(image_id)
                    image_filename = mode + "2014/COCO_"+mode+"2014_"+str(image_id).zfill(12)+".jpg"
                    img = Image.open(self.filebase + image_filename)
                    img = img.convert('RGB')
                    img = transform(img)
                    imgs.append(img)
                    image_ids.append(image_id)
                imgs = torch.stack(imgs)
                imgs = imgs.to(device)
                imgs_feature = extractor(imgs)
                imgs_feature = imgs_feature.cpu()
                for j in range(len(image_ids)):
                    image_id = image_ids[j]
                    img_feature = torch.squeeze(imgs_feature[j])
                    img_feature_filename = 'preprocessed/img_vgg16feature_'+mode+'/'+str(image_id).zfill(12)+'.pt'
                    torch.save(img_feature, img_feature_filename)
            print(mode+' image features preprocessed.')
        
class RandomSampler:
    def __init__(self,data_source,batch_size):
        #self.lengths = [ex[2] for ex in data_source.examples] #sasaki
        self.lengths = [len(i['question']) for i in data_source.data_encoded]
        self.batch_size = batch_size

    def randomize(self):
        #random.shuffle(
        N = len(self.lengths)
        self.ind = np.arange(0,len(self.lengths))
        np.random.shuffle(self.ind)
        self.ind = list(self.ind)
        self.ind.sort(key = lambda x: self.lengths[x])
        self.block_ids = {}
        random_block_ids = list(range(N))
        np.random.shuffle(random_block_ids)
        #generate a random number between 0 to N - 1
        blockid = random_block_ids[0]
        self.block_ids[self.ind[0]] = blockid
        running_count = 1 
        for ind_it in range(1,N):
            if running_count >= self.batch_size or self.lengths[self.ind[ind_it]] != self.lengths[self.ind[ind_it-1]]:
                blockid = random_block_ids[ind_it]
                running_count = 0 
            #   
            self.block_ids[self.ind[ind_it]] = blockid
            running_count += 1
        #  
        # Pdb().set_trace()
        self.ind.sort(key = lambda x: self.block_ids[x])
         

    def __iter__(self):
        # Pdb().set_trace()
        self.randomize()
        return iter(self.ind)

    def __len__(self):
        return len(self.ind)

class VQABatchSampler:
    def __init__(self, data_source, batch_size, drop_last=False):
        #self.lengths = [ex[2] for ex in data_source.examples] #sasaki
        self.lengths = [len(i['question']) for i in data_source.data_encoded]
        # TODO: Use a better sampling strategy.
        # self.sampler = torch.utils.data.sampler.SequentialSampler(data_source)
        self.sampler = RandomSampler(data_source,batch_size)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.unk_emb = 1000

    def __iter__(self):
        batch = []
        prev_len = -1
        this_batch_counter = 0
        for idx in  self.sampler:
            #if self.data_source.examples[idx][4] == self.unk_emb: # sasaki
            #    continue
            #
            curr_len = self.lengths[idx]
            if prev_len > 0 and curr_len != prev_len:
                yield batch
                batch = []
                this_batch_counter = 0
            #
            batch.append(idx)
            prev_len = curr_len
            this_batch_counter += 1
            if this_batch_counter == self.batch_size:
                yield batch
                batch = []
                prev_len = -1
                this_batch_counter = 0
        #
        if len(batch) > 0 and not self.drop_last:
            yield batch
            #self.sampler.randomize()
            prev_len = -1
            this_batch_counter = 0

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

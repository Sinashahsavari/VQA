# feature extaction from pretrained model: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
import torch
import torch.nn as nn
import torchvision.models as models
import utils

from IPython.core.debugger import Pdb

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        x = x / x.norm(p=self.p, dim=1, keepdim=True)
        return x


class ImageEmbedding(nn.Module):
    def __init__(self, image_channel_type='I', output_size=1024):
        super(ImageEmbedding, self).__init__()

        # feature extraction is done in preprocessing

        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

    def forward(self, image, image_ids):

        # feature extraction is done in preprocessing

        image_embedding = self.fflayer(image)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, output_size=1024, num_layers=2, batch_first=True):
        super(QuesEmbedding, self).__init__()
        # TODO: take as parameter
        self.bidirectional = True
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                batch_first=batch_first, bidirectional=self.bidirectional)

            if self.bidirectional:
                self.fflayer = nn.Sequential(
                    nn.Linear(2 * num_layers * hidden_size, output_size),
                    nn.Tanh())
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=batch_first)
            self.fflayer = nn.Sequential(
                nn.Linear(2 * num_layers * hidden_size, output_size),
                nn.Tanh())

    def forward(self, ques):
        _, hx = self.lstm(ques)
        lstm_embedding = torch.cat([hx[0], hx[1]], dim=2)
        ques_embedding = lstm_embedding[0]
        if self.lstm.num_layers > 1 or self.bidirectional:
            for i in range(1, self.lstm.num_layers):
                ques_embedding = torch.cat([ques_embedding, lstm_embedding[i]], dim=1)
            ques_embedding = self.fflayer(ques_embedding)
        return ques_embedding


class VQAModel(nn.Module):

    def __init__(self, vocab_size=10000, word_emb_size=300, emb_size=1024, output_size=1000, image_channel_type='I'):
        super(VQAModel, self).__init__()
        self.word_emb_size = word_emb_size
        self.image_channel = ImageEmbedding(image_channel_type, output_size=emb_size)

        # NOTE the padding_idx below.
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.ques_channel = QuesEmbedding(input_size=word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
        # the original model has "deeplstm" option

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, 1000),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(1000, output_size))
        # the original model has "use_mutan" option

    def forward(self, images, questions, image_ids):
        image_embeddings = self.image_channel(images, image_ids)
        embeds = self.word_embeddings(questions)
        ques_embeddings = self.ques_channel(embeds)
        combined = image_embeddings * ques_embeddings # the original model has "mutan" option
        output = self.mlp(combined)
        return output

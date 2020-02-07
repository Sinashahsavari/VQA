from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import numpy as np



class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


    
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


    
class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits




class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(num_hid):
    w_emb = WordEmbedding(19901, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, 3129, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

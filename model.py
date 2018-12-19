import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, attention=None):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        if attention == None:
            self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.decoder = nn.LSTM(hidden_dim*2, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.v = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim), mean = 0, std = 0.0001))
        self.w1 = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim, hidden_dim), mean=0, std = 0.0001))
        self.w2  = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim, hidden_dim), mean=0, std = 0.0001))
        self.wa  = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim, hidden_dim), mean=0, std = 0.0001))
        self.attention = attention


    def add_attention(self, hidden, encoded):
        hidden = hidden.squeeze()
        a = torch.dot(self.v,
                      torch.nn.functional.relu(torch.mv(self.w1,hidden) + torch.mv(self.w2,encoded)))
        return a

    def mul_attention(self, hidden, encoded):
        hidden = hidden.squeeze()
        a = torch.dot(hidden, torch.mv(self.wa, encoded))
        return a

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def forward(self, input_seq, gold_seq=None):
        input_vectors = self.embeds(torch.tensor(input_seq))
        input_vectors = input_vectors.unsqueeze(1)
        e_outputs, hidden = self.encoder(input_vectors)

        # Technique used to train RNNs:
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        teacher_force = False

        # This condition tells us whether we are in training or inference phase
        if gold_seq is not None and teacher_force:
            gold_vectors = torch.cat([torch.zeros(1, self.hidden_dim), self.embeds(torch.tensor(gold_seq[:-1]))], 0)
            gold_vectors = gold_vectors.unsqueeze(1)
            gold_vectors = torch.nn.functional.relu(gold_vectors)
            outputs, hidden = self.decoder(gold_vectors, hidden)
            predictions = self.out(outputs)
            predictions = predictions.squeeze()
            vals, idxs = torch.max(predictions, 1)
            return predictions, list(np.array(idxs))
        elif self.attention == None:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)
                outputs, hidden = self.decoder(prev, hidden)
                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor([idx]))
                prev = prev.unsqueeze(1)
            return torch.stack(predictions), predicted_seq
        else:
            hidden = torch.zeros(1, 1, self.hidden_dim)
            hidden = (hidden, hidden)
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                e_outputs = e_outputs.squeeze()
                a = torch.zeros(len(input_seq), 1)
                if self.attention == 'add':
                    for j in range(len(input_seq)):
                        a[j] = (self.add_attention(hidden[0], e_outputs[j]))
                else:
                    for j in range(len(input_seq)):
                        a[j] = (self.mul_attention(hidden[0], e_outputs[j]))
                a = torch.nn.functional.softmax(a,dim=0)
                content_vector = torch.zeros(1, 1, self.hidden_dim)
                for j in range(len(input_seq)):
                    content_vector += (a[j]*e_outputs[j])
                concat = torch.cat([prev, content_vector],2)
                concat = torch.nn.functional.relu(concat)
                outputs, hidden = self.decoder(concat, hidden)
                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor([idx]))
                prev = prev.unsqueeze(1)

            return torch.stack(predictions), predicted_seq

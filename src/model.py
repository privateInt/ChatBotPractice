import os
import math
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


from tqdm import tqdm
from tqdm import trange

from torchcrf import CRF

from torch.nn import Transformer
from torch.autograd import Variable 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    '''https://radimrehurek.com/gensim/models/callbacks.html'''
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

class MakeEmbed:
    '''https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec'''
    '''https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#online-training-resuming-training'''
    def __init__(self):
        self.model_dir = "./"
        self.vector_size = 300 # 임베딩 사이즈
        self.window_size = 3   # 몇개의 단어로 예측을 할것인지
        self.workers = 8       # 학습 스레드의 수
        self.min_count = 2     # 단어의 최소 빈도수 (해당 수 미만은 버려진다)
        self.iter = 1000       # 1epoch당 학습 수
        self.sg = 1            # 1: skip-gram, 0: CBOW
        self.model_file = "./data/pretraining/word2vec_skipgram_{}_{}_{}".format(self.vector_size, self.window_size, self.min_count)
        self.epoch_logger = EpochLogger()

    def word2vec_init(self):
        self.word2vec = Word2Vec(size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count,
                         compute_loss=True,
                         iter=self.iter)

    def word2vec_build_vocab(self, dataset):
        self.word2vec.build_vocab(dataset)
        
    def word2vec_most_similar(self, query):
        print(self.word2vec.most_similar(query))
        
    def word2vec_train(self,embed_dataset, epoch = 0, chitchat = False):
        if(epoch == 0):
            epoch = self.word2vec.epochs + 1
        self.word2vec.train(
            sentences=embed_dataset,
            total_examples=self.word2vec.corpus_count,
            epochs=epoch,
            callbacks=[self.epoch_logger]
        )
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
        if(not chitchat):
            self.word2vec.save(self.model_file + '.gensim')
        else:
            self.word2vec.save(self.model_file + '_chitchat.gensim')
        self.vocab = self.word2vec.wv.index2word
        self.vocab = {word: i for i, word in enumerate(self.vocab)}
        
    def load_word2vec(self, chitchat = False):

        if not os.path.exists(self.model_file+'.gensim'):
            raise Exception("모델 로딩 실패 "+ self.model_file+'.gensim')

  
        self.word2vec = Word2Vec.load(self.model_file+'.gensim')
        self.vocab = self.word2vec.wv.index2word
        if(chitchat):
            self.vocab.insert(0,"<S>")
            self.vocab.insert(0,"</S>")
        self.vocab.insert(0,"<UNK>")
        self.vocab.insert(0,"<PAD>")

        self.vocab = {word: i for i, word in enumerate(self.vocab)}
        
    def query2idx(self, query):
        sent_idx = []

        for word in query:
            if(self.vocab.get(word)):
                idx = self.vocab[word]
            else:
                idx = 1

            sent_idx.append(idx)

        return sent_idx

'''
Convolutional Neural Networks for Sentence Classification
Yoon Kim, New York University
'''
class textCNN(nn.Module):
    
    def __init__(self, w2v, dim, kernels, dropout, num_class):
        super(textCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        vocab_size = w2v.size()[0]
        emb_dim = w2v.size()[1]
        self.embed = nn.Embedding(vocab_size+2, emb_dim)
        self.embed.weight[2:].data.copy_(w2v)
        self.embed.weight.requires_grad = False
        
        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim, (w, emb_dim)) for w in kernels])
        #Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        #FC layer
        self.fc = nn.Linear(len(kernels)*dim, num_class)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit

'''
Deep Unordered Composition Rivals Syntactic Methods for Text Classification
Iyyer
'''
class DAN(nn.Module):
    
    def __init__(self, w2v, dim, dropout, num_class = 2):
        super(DAN, self).__init__()
        #load pretrained embedding in embedding layer.
        vocab_size = w2v.size()[0]
        emb_dim = w2v.size()[1]
        self.embed = nn.Embedding(vocab_size+2, emb_dim)
        self.embed.weight[2:].data.copy_(w2v)
        #self.embed.weight.requires_grad = False
        
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.fc1 = nn.Linear(emb_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, num_class)
        
    def forward(self, x):
        emb_x = self.embed(x)
        x = emb_x.mean(dim=1)

        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        logit = self.fc2(x)
        return logit
    
class BiLSTM_CRF(nn.Module):
    '''
    paper : Bidirectional LSTM-CRF Models for Sequence Tagging, Zhiheng Huang,Wei Xu,Kai Yu
    example : https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    '''
    def __init__(self, w2v, tag_to_ix, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = w2v.size()[1]
        self.hidden_dim = hidden_dim
        self.vocab_size =  w2v.size()[0]
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size
        self.START_TAG = "<START_TAG>"
        self.STOP_TAG = "<STOP_TAG>"
        
        self.word_embeds = nn.Embedding(self.vocab_size+2, self.embedding_dim)
        self.word_embeds.weight[2:].data.copy_(w2v)
        self.word_embeds.weight.requires_grad = False
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2, batch_first=True,
                            num_layers=1, bidirectional=True)
    
        # LSTM의 출력을 태그 공간으로 대응시킵니다.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
    
        self.hidden = self.init_hidden()
        
        self.crf = CRF(self.tagset_size, batch_first=True)
        
    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def forward(self, sentence):
        # Bi-LSTM으로부터 배출 점수를 얻습니다.
        self.batch_size = sentence.size()[0]
        self.hidden = self.init_hidden()
        #print(len(sentence))
        embeds = self.word_embeds(sentence)#.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats
    
    def decode(self, logits, mask):
        """
        Viterbi Decoding의 구현체입니다.
        CRF 레이어의 출력을 prediction으로 변형합니다.
        :param logits: 모델의 출력 (로짓)
        :param mask: 마스킹 벡터
        :return: 모델의 예측 (prediction)
        """

        return self.crf.decode(logits, mask)
    
    def compute_loss(self, label, logits, mask):
        """
        학습을 위한 total loss를 계산합니다.
        :param label: label
        :param logits: logits
        :param mask: mask vector
        :return: total loss
        """

        log_likelihood = self.crf(logits, label, mask=mask, reduction='mean')
        return - log_likelihood  # nll loss
    
class Tformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, dff, num_layers, dropout_p=0.5):
        super(Tformer, self).__init__()
        self.transformer = Transformer(dim_model, num_heads, dim_feedforward=dff, num_encoder_layers=num_layers, num_decoder_layers=num_layers,dropout=dropout_p)
        self.pos_encoder = PositionalEncoding(dim_model, dropout_p)
        self.encoder = nn.Embedding(num_tokens, dim_model)

        self.pos_encoder_d = PositionalEncoding(dim_model, dropout_p)
        self.encoder_d = nn.Embedding(num_tokens, dim_model)

        self.dim_model = dim_model
        self.num_tokens = num_tokens

        self.linear = nn.Linear(dim_model, num_tokens)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, srcmask, tgtmask, srcpadmask, tgtpadmask):
        src = self.encoder(src) * math.sqrt(self.dim_model)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt) * math.sqrt(self.dim_model)
        tgt = self.pos_encoder_d(tgt)

        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), srcmask, tgtmask, src_key_padding_mask=srcpadmask, tgt_key_padding_mask=tgtpadmask)
        output = self.linear(output)
        return output
    
    def gen_attention_mask(self, x):
        mask = torch.eq(x, 0)
        return mask
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)
    
def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)
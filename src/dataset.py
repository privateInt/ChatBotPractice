import os
import sys
import json
import torch
import random

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch.utils.data as data

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class ChitChatDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor, labels):
        super(ChitChatDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        self.labels = labels
        
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.labels[index]

    def __len__(self):
        return len(self.x)
    
class EntityDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor, lengths):
        super(EntityDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        self.lengths = lengths
        
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index]

    def __len__(self):
        return len(self.x)
    
class Preprocessing:
    '''
    데이터의 최대 token길이가 10이지만
    실제 환경에서는 얼마의 길이가 들어올지 몰라 적당한 길이 부여
    '''
    
    def __init__(self, max_len = 20):
        self.max_len = max_len
        self.PAD = 0
        self.O = 0
        
    def pad_idx_sequencing(self, q_vec):
        q_len = len(q_vec)
        diff_len = q_len - self.max_len
        if(diff_len>0):
            q_vec = q_vec[:self.max_len]
            q_len = self.max_len
        else:
            pad_vac = [0] * abs(diff_len)
            q_vec += pad_vac

        return q_vec
    
    def make_batch(self):
        pass

class MakeDataset:
    def __init__(self, pos_flag = False):
        
        self.intent_label_dir = "./data/dataset/intent_label.json"
        self.intent_ood_label_dir = "./data/dataset/intent_label_with_ood.json"
        self.entity_label_dir = "./data/dataset/entity_label.json"
        self.intent_data_dir = "./data/dataset/intent_data.csv"
        self.entity_data_dir = "./data/dataset/entity_data.csv"
        self.ood_data_dir = "./data/dataset/ood_data.csv"
        self.chitchat_data_dir = "./data/dataset/ChatbotData.csv"
        
        self.pos = pos_flag
        if(self.pos):
            self.okt = Okt()
        self.entity_label = self.load_entity_label()
        self.intent_label = self.load_intent_label()
        self.prep = Preprocessing()
        vocab_file = "./data/pretraining/chatbot.model"
        self.transformers_tokenizer = spm.SentencePieceProcessor()
        self.transformers_tokenizer.load(vocab_file)
        
    def load_entity_label(self):
        f = open(self.entity_label_dir, encoding="UTF-8")
        entity_label = json.loads(f.read())
        self.entitys = list(entity_label.keys())
        return entity_label
    
    def load_intent_label(self):
        f = open(self.intent_label_dir, encoding="UTF-8")
        intent_label = json.loads(f.read())
        self.intents = list(intent_label.keys())
        return intent_label
    
    def tokenize(self, sentence):
        if(self.pos):
            pass            
        else:
            return sentence.split()
    
    def tokenize_dataset(self, dataset):
        token_dataset = []
        for data in dataset:
            token_dataset.append(self.tokenize(data))
        return token_dataset
    
    def make_embed_dataset(self, ood = False):
        embed_dataset = pd.read_csv(self.intent_data_dir)
        if(ood):
            ood_dataset = pd.read_csv(self.ood_data_dir)
            embed_dataset = pd.concat([embed_dataset, ood_dataset])
        embed_dataset = embed_dataset["question"].to_list()
        embed_dataset = self.tokenize_dataset(embed_dataset)
        return embed_dataset
    
    def encode_dataset(self, dataset):
        token_dataset = []
        for data in dataset:
            token_dataset.append( [2] + self.transformers_tokenizer.encode_as_ids(data) + [3])
        return token_dataset

    def make_chitchat_dataset(self, train_ratio = 0.8):
        chitchat_dataset = pd.read_csv(self.chitchat_data_dir)
        Qs = chitchat_dataset["Q"].tolist()
        As = chitchat_dataset["A"].tolist()
        label = chitchat_dataset["label"].tolist()
        
        Qs = self.encode_dataset(Qs)
        As = self.encode_dataset(As)
        
        self.prep.max_len = 40
        x, y = [], []
        for q, a in zip(Qs,As):
            x.append(self.prep.pad_idx_sequencing(q))
            y.append(self.prep.pad_idx_sequencing(a))
        x = torch.tensor(x)
        y = torch.tensor(y)
        x_len = x.size()[0]
        train_size = int(x_len*train_ratio)
        
        if(train_ratio == 1.0):
            train_x = x[:train_size]
            train_y = y[:train_size]
            train_label = label[:train_size]
            train_dataset = ChitChatDataset(train_x,train_y,train_label)
            return train_dataset, None
        else:
            train_x = x[:train_size]
            train_y = y[:train_size]
            train_label = label[:train_size]

            test_x = x[train_size+1:]
            test_y = y[train_size+1:]
            test_label = label[train_size+1:]

            train_dataset = ChitChatDataset(train_x,train_y,train_label)
            test_dataset = ChitChatDataset(test_x,test_y,test_label)

            return train_dataset, test_dataset
    
    def make_entity_dataset(self, embed):
        entity_dataset = pd.read_csv(self.entity_data_dir)
        entity_querys = self.tokenize_dataset(entity_dataset["question"].tolist())
        labels = []
        for label in entity_dataset["label"].to_list():
            temp = []
            for entity in label.split():
                temp.append(self.entity_label[entity])
            labels.append(temp)
        dataset = list(zip(entity_querys, labels))
        entity_train_dataset, entity_test_dataset = self.word2idx_dataset(dataset, embed)
        return entity_train_dataset, entity_test_dataset

    def make_intent_dataset(self, embed):
        intent_dataset = pd.read_csv(self.intent_data_dir)
        intent_querys = self.tokenize_dataset(intent_dataset["question"].tolist())
        labels = [self.intent_label[label] for label in intent_dataset["label"].to_list()]
        dataset = list(zip(intent_querys, labels))
        intent_train_dataset, intent_test_dataset = self.word2idx_dataset(dataset, embed)
        return intent_train_dataset, intent_test_dataset
    
    def make_ood_dataset(self, embed):
        intent_dataset = pd.read_csv(self.intent_data_dir)
        ood_dataset = pd.read_csv(self.ood_data_dir).sample(frac=1).reset_index(drop=True)
        intent_dataset = pd.concat([intent_dataset,ood_dataset])
        labels = []
        for label in intent_dataset["label"].to_list():
            if(label == "OOD"):
                labels.append(0)
            else:
                labels.append(1)
            
        intent_querys = self.tokenize_dataset(intent_dataset["question"].tolist())
        
        dataset = list(zip(intent_querys, labels))
        intent_train_dataset, intent_test_dataset = self.word2idx_dataset(dataset, embed)
        return intent_train_dataset, intent_test_dataset
    
    def word2idx_dataset(self, dataset ,embed, train_ratio = 0.8):
        embed_dataset = []
        question_list, label_list, lengths = [], [], []
        flag = True
        random.shuffle(dataset)
        for query, label in dataset :
            q_vec = embed.query2idx(query)
            lengths.append(len(q_vec))
            
            q_vec = self.prep.pad_idx_sequencing(q_vec)
            #print(label)
            question_list.append(torch.tensor([q_vec]))
            if(isinstance(label, list)):
                label = self.prep.pad_idx_sequencing(label)
                label_list.append(label)
                flag = False
            else:
                label_list.append(torch.tensor([label]))

        x = torch.cat(question_list)
        if(flag):
            y = torch.cat(label_list)
        else:
            y = torch.tensor(label_list)

        x_len = x.size()[0]
        y_len = y.size()[0]
        if(x_len == y_len):
            train_size = int(x_len*train_ratio)
            
            train_x = x[:train_size]
            train_y = y[:train_size]

            test_x = x[train_size+1:]
            test_y = y[train_size+1:]
            if(flag):
                train_dataset = TensorDataset(train_x,train_y)
                test_dataset = TensorDataset(test_x,test_y)
            else:
                train_length = lengths[:train_size]
                test_length = lengths[train_size+1:]
            
                train_dataset = EntityDataset(train_x,train_y,train_length)
                test_dataset = EntityDataset(test_x,test_y,test_length)
            
            return train_dataset, test_dataset
            
        else:
            print("ERROR x!=y")
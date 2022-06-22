import os
import sys
import json
import torch
import random

from src.model import BiLSTM_CRF, MakeEmbed, textCNN, DAN, EpochLogger, save
from src.dataset import Preprocessing, MakeDataset

class NaturalLanguageUnderstanding:
    
    def __init__(self):
        self.dataset = MakeDataset()
        self.embed = MakeEmbed()
        self.embed.load_word2vec()

        self.weights = self.embed.word2vec.wv.vectors
        self.weights = torch.FloatTensor(self.weights)
        
        self.intent_clsf = textCNN(self.weights, 256, [3,4,5], 0.5, len(self.dataset.intent_label))
        self.slot_tagger = BiLSTM_CRF(self.weights, self.dataset.entity_label, 256, 128)
        self.ood_detector = DAN(self.weights, 256, 0.5, 2)
        
    def init_NLU_result(self):
        NLU_result = {
                    "INTENT" : "",
                    "SLOT"   :[
            
                        ]
                    }
        return NLU_result
    
    def model_load(self, intent_path, slot_path, ood_path):
        self.intent_clsf.load_state_dict(torch.load(intent_path))
        self.slot_tagger.load_state_dict(torch.load(slot_path))
        self.ood_detector.load_state_dict(torch.load(ood_path))
        self.intent_clsf.eval()
        self.slot_tagger.eval()
        self.ood_detector.eval()
        
    def predict(self, query):
        x = self.dataset.prep.pad_idx_sequencing(self.embed.query2idx(self.dataset.tokenize(query)))

        x = torch.tensor(x)
        '''
        ood dectector
        '''
        f = self.ood_detector(x.unsqueeze(0))
        ood = torch.argmax(f).tolist()
        if(ood):
            '''
            intent clsf
            '''
            f = self.intent_clsf(x.unsqueeze(0))

            intent = self.dataset.intents[torch.argmax(f).tolist()]
        else:
            intent = "ood"

        '''
        slot tagger
        '''
        f = self.slot_tagger(x.unsqueeze(0))

        mask = torch.where(x > 0, torch.tensor([1.]), torch.tensor([0.])).type(torch.uint8)

        predict = self.slot_tagger.decode(f,mask.view(1,-1))
        return intent, predict
    
    def convert_nlu_result(self, query, intent, predict):
        NLU_result = self.init_NLU_result()
        x_token =query.split()

        slots = []
        BIE = []
        prev = "";
        for i, slot in enumerate([self.dataset.entitys[p] for p in predict[0]]):
            name = slot[2:]

            if("S-" in slot):
                if(BIE != []):
                    slots.append(prev[2:] +"^"+" ".join(BIE))
                    BIE = []
                slots.append(name+"^"+x_token[i])
            elif("B-" in slot):
                BIE.append(x_token[i])
                prev = slot
            elif("I-" in slot and "B" in prev):
                BIE.append(x_token[i])
                prev = slot
            elif("E-" in slot and ("I" in prev or "E" in prev)):
                BIE.append(x_token[i])
                slots.append(name+"^"+" ".join(BIE))
                BIE = []
            else:
                if(BIE != []):
                    slots.append(prev[2:]+"^"+" ".join(BIE))
                    BIE = []
        NLU_result["INTENT"] = intent
        NLU_result["SLOT"]   = slots
        return NLU_result
    
    def run(self, query):
        intent, predict = self.predict(query)
        self.nlu_predict = [intent, predict]
        NLU_result = self.convert_nlu_result(query, intent, predict)
        return NLU_result
    def print_nlu_result(self, nlu_result):
        print('발화 의도 : ' + nlu_result.get('INTENT'))
        print('발화 개체 : ')
        for slot_concat in nlu_result.get('SLOT'):
            slot_name = slot_concat.split('^')[0]
            slot_value = slot_concat.split('^')[1]
            print("    "+slot_name + " : " + slot_value)
import os
import re
import sys
import json
import random
import torch
import pandas as pd
import numpy as np
from collections import deque
from src.model import BiLSTM_CRF, MakeEmbed, textCNN, EpochLogger, Tformer, save
from src.dataset import Preprocessing, MakeDataset
from src.NLU import NaturalLanguageUnderstanding

class DialogSearch:
    
    def __init__(self):
        self.template_dir = "./data/search_template_dataset.csv"
        self.values = {
                    "DATE" : "",
                    "LOCATION" : "",
                    "PLACE" : "",
                    "RESTAURANT" : ""
                 }
        
        self.template = self.load_template()

    def load_template(self):
        template = pd.read_csv(self.template_dir)
        return template
    
    def search_template(self, nlu_result):
        intent, slots = self.make_search_key(nlu_result)
        matched_template = []
        print("")
        print("#####대화 흐름 검색 시작######")
        for data in self.template.iterrows():
            intent_flag = False
            slot_flag = False
            
            row = data[1]
            if(row["label"] == intent):
                intent_flag = True
            if(isinstance(row.get("slot"), str)):
                template_slots = sorted(row["slot"].split("^"))
                key_slots = sorted(slots.split("^"))
                if(template_slots == key_slots):
                    slot_flag = True
            elif(slots == ""):
                slot_flag = True
            
            if(intent_flag and slot_flag):
                print("#############매칭#############")
                matched_template.append(row["template"])
        print("#####대화 흐름 검색 종료######")
        print("")
        return matched_template
    
    def make_search_key(self, nlu_result):
        intent = nlu_result.get("INTENT")

        keys = set()
        for name_value in nlu_result.get("SLOT"):
            slot_name = name_value.split("^")[0]
            slot_value = name_value.split("^")[1]
            keys.add(slot_name)
            self.values[slot_name] = slot_value

        slots = "^".join(keys)
        return intent, slots
    
    def replace_slot(self, flag, key, template):
        value = self.values.get(key)
        key = "{"+key+"}"
        if(value != ""):
            template = template.replace(key,value)
        else:
            template = ""
        flag = not flag
        return flag, template
    
    def filling_NLG_slot(self, templates):
        filling_templates = []

        for template in templates:
            date_index = template.find("{DATE}")
            location_index = template.find("{LOCATION}")
            place_index = template.find("{PLACE}")
            restraurant_index = template.find("{RESTAURANT}")

            date_flag = date_index == -1
            location_flag = location_index == -1
            place_flag = place_index == -1
            restraurant_flag = restraurant_index == -1
            
            cnt = 0
            while(not (date_flag and location_flag and place_flag and restraurant_flag)):
                print("before : "+template)
                if(not date_flag):
                    key = "DATE"
                    date_flag, template = self.replace_slot(date_flag, key, template)

                if(not location_flag):
                    key = "LOCATION"
                    location_flag, template = self.replace_slot(location_flag, key, template)

                if(not place_flag):
                    key = "PLACE"
                    date_flag, template = self.replace_slot(place_flag, key, template)

                if(not restraurant_flag):
                    key = "RESTAURANT"
                    location_flag, template = self.replace_slot(restraurant_flag, key, template)
                print("after : "+template)
                filling_templates.append(template)
        
        return filling_templates
    
    def select_template(self, templates):
        template_size = len(templates)
        if(template_size == 1):
            template = templates[0]
        elif(template_size > 1):
            template = random.choice(templates)
        else:
            template = ""
        return template

    def run_search(self, nlu_result):
        templates = self.search_template(nlu_result)
        print("#######템플릿 매칭 시작#######")
        for i, template in enumerate(templates):
            print(str(i)+". template : "+template)
        print("#######템플릿 매칭 종료#######")
        print("")
        print("######템플릿 채우기 시작######")
        templates = self.filling_NLG_slot(templates)
        print("######템플릿 채우기 종료######")
        print("")
        template = self.select_template(templates)
        if(template == ""):
            template == "죄송합니다. 다시 말해주세요."
        return template
    
class StateConditionCheck:
    def __init__(self, s_state, e_state, cond):
        self.s_state = s_state
        self.e_state = e_state
        self.conds = cond
        self.parse_cond()
    def print_all(self):
        print("s:{0}, e:{1}, cond:{2}".format(self.s_state,self.e_state,self.conds))
    def parse_cond(self):
        self.conds = self.conds.split()
    def print_end_state(self):
        return self.e_state
    def cond_check(self, nlu_slot):
        conds_check = True
        cond_score = 0
        for cond in self.conds:
            if("==" in cond):
                cond = cond.split("==")
                if(len(cond) == 2):
                    right = cond[0]
                    left  = cond[1]
                    if(right == "prev_ans_state"):
                        if(left in nlu_slot):
                            conds_check = conds_check and True
                            cond_score += 1
                        else:
                            return False , 0
                else:
                    return False , 0
            else:
                cond = cond.split("_")
                if(len(cond) == 1):
                    if(cond[0] == "PASS"):
                        return True, 1
                if(len(cond) == 2):
                    operator = cond[0]
                    value = cond[1]
                    if(operator == "EX"):
                        if(value in nlu_slot):
                            conds_check = conds_check and True
                            cond_score += 1
                        else:
                            return False , 0
        return conds_check,cond_score
    
class DialogManager:
    
    def __init__(self):
        self.template_dir = "./data/template_dataset.csv"
        self.plan_dir = "./data/plan.csv"
        self.values = {
                    "DATE" : "",
                    "LOCATION" : "",
                    "PLACE" : "",
                    "RESTAURANT" : ""
                 }
        
        self.graph = self.load_plan()
        self.state_transition = []
        self.start_state = "DS_START"
        self.dm_result = {
            "STATE": "",
            "SLOT" : "",
            "NLU"  : {},
            "NLG"  : []
        }
        self.prev_ans_state = "DS_START"
        self.nlg = NLG()
    def __dm_result_init__(self):
        self.dm_result = {
            "STATE": "",
            "SLOT" : "",
            "NLU"  : {},
            "NLG"  : []
        }
    def load_plan(self):
        graph = {}
        plan = pd.read_csv(self.plan_dir)
        for rows in plan.iterrows():
            s_state = rows[1]["START STATE"]
            e_state = rows[1]["END STATE"]
            cond = rows[1]["CONDITION"]
            if(graph.get(s_state)):
                graph[s_state].append(StateConditionCheck(s_state,e_state,cond))
            else:
                graph[s_state] = [StateConditionCheck(s_state,e_state,cond)]
        return graph

    def tracking(self, start, visited, dm_slot):
        queue = deque([start])
        visited.append(start)
        score = {}
        count = 0
        while queue:
            req_flag = False
            cur = queue.popleft()
            if(self.graph.get(cur)):
                for state in self.graph[cur]:
                    cond_check, cond_score = state.cond_check(dm_slot)    
                    if cond_check:
                        state_name = state.print_end_state()
                        #queue.append(state_name)
                        #visited.append(state_name)
                        score[state_name] = cond_score
                #print(score)
                max_cnt = 0
                state_name = "DS_END"
                for name in score:
                    cnt = score[name]
                    if(max_cnt<=cnt):
                        max_cnt = cnt
                        state_name = name
                '''
                #api 실행 함수 run 및 데이터 저장
                '''
                queue.append(state_name)
                visited.append(state_name)
                if("REQ_" in state_name):
                    req_flag = True
                    break

            if(req_flag):
                break
            count+=1
            if(count == 10):
                break
                
    def make_dm_slot(self, nlu_slot):
        dm_slot = []
        dm_slot.append(nlu_slot["INTENT"])
        for slot in nlu_slot["SLOT"]:
            dm_slot.append(slot.split("^")[0])
        dm_slot.append(self.prev_ans_state)
        return dm_slot
    
    def dst(self, nlu_slot):
        #print("#####대화 상태 추적 시작######")
        self.__dm_result_init__()
        dm_slot = self.make_dm_slot(nlu_slot)
        
        self.tracking(self.start_state, self.state_transition, dm_slot)
        self.start_state = self.state_transition[-1]
        
        if("ANS_" in self.start_state):
            self.prev_ans_state = self.start_state
            self.dm_result["STATE"] = self.start_state
            self.dm_result["SLOT"]  = nlu_slot["SLOT"]
            self.dm_result["NLU"]   = nlu_slot
            response = self.nlg.run(self.dm_result)
            self.dm_result["NLG"].append(response)
            
            self.start_state = "DS_START"
            self.tracking(self.start_state, self.state_transition, dm_slot)
            self.start_state = self.state_transition[-1]
            self.dm_result["STATE"] = self.start_state

        elif("REQ_" in self.start_state):
            self.dm_result["STATE"] = self.start_state
            self.dm_result["SLOT"]  = nlu_slot["SLOT"]
            self.dm_result["NLU"]   = nlu_slot
            response = self.nlg.run(self.dm_result)
            self.dm_result["NLG"].append(response)
        else:
            self.start_state = "DS_REQ_USER_INPUT"
        #print("#####대화 상태 추적 종료######")
        return self.dm_result
    def run(self, nlu_slot):
        dm_result = self.dst(nlu_slot)
        return dm_result
    
    def clear(self):
        self.state_transition = []
        self.start_state = "DS_START"

class NLG:
    
    def __init__(self):
        self.template_dir = "./data/template_dataset.csv"
        self.values = {
                    "DATE" : "",
                    "LOCATION" : "",
                    "PLACE" : "",
                    "RESTAURANT" : ""
                 }
        
        self.template = self.load_template()

    def load_template(self):
        template = pd.read_csv(self.template_dir).fillna("")
        return template
    
    def search_template(self, dm_result):
        state, slots = self.make_search_key(dm_result)
        matched_template = []
        #print("")
        #print("#######템플릿 검색 시작#######")
        for data in self.template.iterrows():
            state_flag = False
            slot_flag = False
            
            row = data[1]
            if(row["state"] == state):
                state_flag = True
            if(isinstance(row.get("slot"), str)):
                if(row["slot"] == ""):
                    slot_flag = True
                else:
                    template_slots = sorted(row["slot"].split("^"))
                    key_slots = sorted(slots.split("^"))
                    if(template_slots == key_slots):
                        slot_flag = True
            elif(slots == ""):
                slot_flag = True

            if(state_flag and slot_flag):
                #print("#############매칭#############")
                matched_template.append(row["template"])
        #print("#######템플릿 검색 종료#######")
        #print("")
        return matched_template
    
    def make_search_key(self, dm_result):
        state = dm_result.get("STATE")

        keys = set()
        for name_value in dm_result.get("SLOT"):
            slot_name = name_value.split("^")[0]
            slot_value = name_value.split("^")[1]
            keys.add(slot_name)
            self.values[slot_name] = slot_value

        slots = "^".join(keys)
 
        return state, slots
    
    def replace_slot(self, flag, key, template):
        value = self.values.get(key)
        key = "{"+key+"}"
        if(value != ""):
            template = template.replace(key,value)
        else:
            template = ""
        flag = not flag
        return flag, template
    
    def filling_NLG_slot(self, templates):
        filling_templates = []

        for template in templates:
            date_index = template.find("{DATE}")
            location_index = template.find("{LOCATION}")
            place_index = template.find("{PLACE}")
            restraurant_index = template.find("{RESTAURANT}")

            date_flag = date_index == -1
            location_flag = location_index == -1
            place_flag = place_index == -1
            restraurant_flag = restraurant_index == -1
            if(date_flag and location_flag and place_flag and restraurant_flag):
                filling_templates.append(template)
                break
            cnt = 0
            while(not (date_flag and location_flag and place_flag and restraurant_flag)):
                #print("before : "+template)
                if(not date_flag):
                    key = "DATE"
                    date_flag, template = self.replace_slot(date_flag, key, template)

                if(not location_flag):
                    key = "LOCATION"
                    location_flag, template = self.replace_slot(location_flag, key, template)

                if(not place_flag):
                    key = "PLACE"
                    date_flag, template = self.replace_slot(place_flag, key, template)

                if(not restraurant_flag):
                    key = "RESTAURANT"
                    location_flag, template = self.replace_slot(restraurant_flag, key, template)
                #print("after : "+template)
                filling_templates.append(template)
        
        return filling_templates
    
    def select_template(self, templates):
        template_size = len(templates)
        if(template_size == 1):
            template = templates[0]
        elif(template_size > 1):
            template = random.choice(templates)
        else:
            template = ""
        return template

    def run(self, dm_result):
        templates = self.search_template(dm_result)
        #print("#######템플릿 매칭 시작#######")
        #for i, template in enumerate(templates):
        #    print(str(i)+". template : "+template)
        #print("#######템플릿 매칭 종료#######")
        #print("")
        #print("######템플릿 채우기 시작######")
        templates = self.filling_NLG_slot(templates)
        #print("######템플릿 채우기 종료######")
        #print("")
        template = self.select_template(templates)
        return template
    
class E2E_dialog:
    def __init__(self, dataset, model_path):
        self.vocab = dataset.transformers_tokenizer
        self.vocab_size = dataset.transformers_tokenizer.vocab_size()
        
        self.model = Tformer(num_tokens=self.vocab_size, dim_model=256, num_heads=8, dff=512, num_layers=2, dropout_p=0.1)
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.MAX_LENGTH = 50
        
    def preprocess_sentence(self, sentence):
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        return sentence

    def evaluate(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        input = torch.tensor([[2] + self.vocab.encode_as_ids(sentence) + [3]])
        output = torch.tensor([[2]])

        # 디코더의 예측 시작
        ps = []
        for i in range(self.MAX_LENGTH):
            src_mask = self.model.generate_square_subsequent_mask(input.shape[1])
            tgt_mask = self.model.generate_square_subsequent_mask(output.shape[1])

            src_padding_mask = self.model.gen_attention_mask(input)
            tgt_padding_mask = self.model.gen_attention_mask(output)

            predictions = self.model(input, output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).transpose(0,1)
            # 현재(마지막) 시점의 예측 단어를 받아온다.
            predictions = predictions[:, -1:, :]
            predictions = torch.softmax(predictions.view(-1).cpu(), dim=0)
            predictions = torch.max(predictions, axis = -1)
            predicted_p = predictions.values
            ps.append(predicted_p)
            predicted_id =predictions.indices.view(1,1)


            # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
            if torch.equal(predicted_id[0][0], torch.tensor(3)):
                break

            # 마지막 시점의 예측 단어를 출력에 연결한다.
            # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
            output = torch.cat([output, predicted_id], axis=1)

        return torch.squeeze(output, axis=0).cpu().numpy(), (sum(ps)/len(ps)).detach().numpy()

    def predict(self, sentence):
        prediction, predicted_sentence_p = self.evaluate(sentence)
        predicted_sentence = self.vocab.Decode(list(map(int,[i for i in prediction if i < self.vocab_size])))

        #print('Input: {}'.format(sentence))
        #print('Output: {}'.format(predicted_sentence))

        return predicted_sentence, float(predicted_sentence_p)
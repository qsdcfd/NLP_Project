""" 토크나이저 확인하기 """
# https://github.com/thunlp/PLMpapers
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

from transformers import RobertaModel
model = RobertaModel.from_pretrained('roberta-base')

""" for CoM """
batch_com_out = model(input_ids=batch_padding_token, attention_mask=batch_padding_attention_mask)['last_hidden_state']
batch_com_final = batch_com_out[:,0,:] # CLS 토큰의 output 가져오기 위해

result = model(input_ids=batch_padding_token, attention_mask=batch_padding_attention_mask)

import torch

model2 = RobertaModel.from_pretrained('roberta-base')
# 발화1: feature1 [1, 768]
# 발화3: feature3 [1, 768]
# 발화6에 해당하는 감정을 예측할 때 발화1, 발화3의 정보를 사용할 것
# feature1 + feature3
# (feature1, feature6) 어텐션 weights w1
# (feature3, feature6) 어텐션 weights w3
# w1*feature1 + w3*feature6
# GRU(feature1, feature3)

""" GRU 세팅 """
import torch.nn as nn 
hiddenDim = model2.config.hidden_size
zero = torch.empty(2, 1, hiddenDim)
h0 = torch.zeros_like(zero) # (num_layers * num_directions, batch, hidden_size)
speakerGRU = nn.GRU(hiddenDim, hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)

""" GRU 통과 --> PM 결과 """
batch_pm_gru_final = []
for PM_inputs in batch_PM_input:
    if PM_inputs:
        pm_outs = []
        for PM_input in PM_inputs:
            pm_out = model2(PM_input)['last_hidden_state'][:,0,:] # CLS의 출력
            pm_outs.append(pm_out)
        pm_outs = torch.cat(pm_outs, 0).unsqueeze(1) # (speaker_num, batch=1, hidden_dim)
        pm_gru_outs, _ = speakerGRU(pm_outs, h0) # (speaker_num, batch=1, hidden_dim)
        pm_gru_final = pm_gru_outs[-1,:,:] # (1, hidden_dim)
        batch_pm_gru_final.append(pm_gru_final)
    else:
        batch_pm_gru_final.append(torch.zeros(1, hiddenDim))
batch_pm_gru_final = torch.cat(batch_pm_gru_final, 0)



## GPU 작동 모델 수정
from transformers import RobertaModel
import torch
import torch.nn as nn

class ERC_model(nn.Module):
    def __init__(self, clsNum):
        super(ERC_model, self).__init__()
        self.com_model = RobertaModel.from_pretrained('roberta-base')
        self.pm_model = RobertaModel.from_pretrained('roberta-base')
        
        """ GRU 세팅 """
        self.hiddenDim = self.com_model.config.hidden_size
        zero = torch.empty(2, 1, self.hiddenDim)
        self.h0 = torch.zeros_like(zero).cuda() # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
        
        """ score matrix """
        self.W = nn.Linear(self.hiddenDim, clsNum)
    def forward(self, batch_padding_token, batch_padding_attention_mask, batch_PM_input):
        """ for CoM """
        batch_com_out = self.com_model(input_ids=batch_padding_token, attention_mask=batch_padding_attention_mask)['last_hidden_state']
        batch_com_final = batch_com_out[:,0,:]
        
        """ GRU 통과 --> PM 결과 """
        batch_pm_gru_final = []
        for PM_inputs in batch_PM_input:
            if PM_inputs:
                pm_outs = []
                for PM_input in PM_inputs:
                    pm_out = self.pm_model(PM_input)['last_hidden_state'][:,0,:]
                    pm_outs.append(pm_out)
                pm_outs = torch.cat(pm_outs, 0).unsqueeze(1) # (speaker_num, batch=1, hidden_dim)
                pm_gru_outs, _ = self.speakerGRU(pm_outs, self.h0) # (speaker_num, batch=1, hidden_dim)
                pm_gru_final = pm_gru_outs[-1,:,:] # (1, hidden_dim)
                batch_pm_gru_final.append(pm_gru_final)
            else:
                batch_pm_gru_final.append(torch.zeros(1, self.hiddenDim).cuda())
        batch_pm_gru_final = torch.cat(batch_pm_gru_final, 0)        
        
        """ score matrix """
        final_output = self.W(batch_com_final + batch_pm_gru_final) # (B, C)
        
        return final_output
      
      
import torch.nn as nn
def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pdb

training_epochs = 5
max_grad_norm = 10
lr = 1e-6
num_training_steps = len(train_dataset)*training_epochs
num_warmup_steps = len(train_dataset)
optimizer = torch.optim.AdamW(erc_model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

for epoch in tqdm(range(training_epochs)):
    erc_model.train() 
    for i_batch, data in enumerate(train_dataloader):
        batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
        batch_padding_token = batch_padding_token.cuda()
        batch_padding_attention_mask = batch_padding_attention_mask.cuda()
        batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
        batch_label = batch_label.cuda()        
        
        """Prediction"""
        pred_logits = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
        
        """Loss calculation & training"""
        loss_val = CELoss(pred_logits, batch_label)
        
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(erc_model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()        
        break
    break
  

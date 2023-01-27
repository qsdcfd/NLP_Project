import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

training_epochs = 5
max_grad_norm = 10
lr = 1e-6
num_training_steps = len(train_dataset)*training_epochs
num_warmup_steps = len(train_dataset)
optimizer = torch.optim.AdamW(erc_model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

best_dev_fscore = 0
save_path = '.'
for epoch in tqdm(range(training_epochs)):
    erc_model.train() 
    for i_batch, data in enumerate(tqdm(train_dataloader)):
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
    
    """Dev & Test evaluation"""
    erc_model.eval()
    
    dev_acc, dev_pred_list, dev_label_list = CalACC(erc_model, dev_dataloader)
    dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
    
    print("Dev W-avg F1: {}".format(dev_fbeta))

    """Best Score & Model Save"""
    if dev_fbeta > best_dev_fscore:
        best_dev_fscore = dev_fbeta

        test_acc, test_pred_list, test_label_list = CalACC(erc_model, test_dataloader)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                

        SaveModel(erc_model, save_path)
        print("Test W-avg F1: {}".format(test_fbeta))

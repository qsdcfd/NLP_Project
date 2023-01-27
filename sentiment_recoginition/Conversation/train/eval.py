import torch
from tqdm import tqdm
def CalACC(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_padding_token, batch_padding_attention_mask, batch_PM_input, batch_label = data
            batch_padding_token = batch_padding_token.cuda()
            batch_padding_attention_mask = batch_padding_attention_mask.cuda()
            batch_PM_input = [[x2.cuda() for x2 in x1] for x1 in batch_PM_input]
            batch_label = batch_label.cuda()        

            """Prediction"""
            pred_logits = erc_model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_label.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)
    return acc, pred_list, label_list
  
from sklearn.metrics import precision_recall_fscore_support
dev_acc, dev_pred_list, dev_label_list = CalACC(erc_model, dev_dataloader)
dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

from dataset import data_loader
from torch.utils.data import DataLoader

train_dataset = data_loader('./MELD/data/MELD/train_sent_emo.csv')
dev_dataset = data_loader('./MELD/data/MELD/dev_sent_emo.csv')
test_dataset = data_loader('./MELD/data/MELD/test_sent_emo.csv')

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dev_dataset.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)

from model import ERC_model
clsNum = len(train_dataset.emoList)
erc_model = ERC_model(clsNum).cuda()

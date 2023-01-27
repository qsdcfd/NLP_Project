import torch
from dataset import data_loader
from torch.utils.data import DataLoader

test_dataset = data_loader('./MELD/data/MELD/test_sent_emo.csv')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)

from model import ERC_model
clsNum = len(test_dataset.emoList)
erc_model = ERC_model(clsNum).cuda()
model_path = './model.bin'
erc_model.load_state_dict(torch.load(model_path))
erc_model.eval()
print('')

from tqdm import tqdm
def ErrorSamples(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []    
    
    error_samples = []
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
            if pred_label != true_label:
                error_samples.append([batch_padding_token, true_label, pred_label])
            if pred_label == true_label:
                correct += 1
        acc = correct/len(dataloader)                
    return error_samples, acc, pred_list, label_list
  
error_samples, acc, pred_list, label_list = ErrorSamples(erc_model, test_dataloader)
  
import random
random_error_samples = random.sample(error_samples, 5)

for random_error_sample in random_error_samples:
    batch_padding_token, true_label, pred_label = random_error_sample
    print('--------------------------------------------------------')
    print("입력 문장들: ", test_dataset.tokenizer.decode(batch_padding_token.squeeze(0).tolist()))
    print("정답 감정: ", test_dataset.emoList[true_label])
    print("예측 감정: ", test_dataset.emoList[pred_label])

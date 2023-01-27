import csv
from torch.utils.data import Dataset
def split(session):
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])    
    return final_data

class data_loader(Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'r')
        rdr = csv.reader(f)
        
        """ 추가 """
        emoSet = set()

        """ 세션 데이터 저장할 것"""
        self.session_dataset = []
        session = []
        speaker_set = []

        """ 실제 데이터 저장 방식 """
        pre_sess = 'start'
        for i, line in enumerate(rdr):
            if i == 0:
                """ 저장할 데이터들 index 확인 """
                header  = line
                utt_idx = header.index('Utterance')
                speaker_idx = header.index('Speaker')
                emo_idx = header.index('Emotion')
                sess_idx = header.index('Dialogue_ID')
            else:
                utt = line[utt_idx]
                speaker = line[speaker_idx]
                """ 유니크한 스피커로 바꾸기 """
                if speaker in speaker_set:
                    uniq_speaker = speaker_set.index(speaker)
                else:
                    speaker_set.append(speaker)
                    uniq_speaker = speaker_set.index(speaker)
                emotion = line[emo_idx]
                sess = line[sess_idx]

                if pre_sess == 'start' or sess == pre_sess:
                    session.append([uniq_speaker, utt, emotion])
                else:
                    """ 세션 데이터 저장 """
                    self.session_dataset += split(session)
                    session = [[uniq_speaker, utt, emotion]]
                    speaker_set = []
                    emoSet.add(emotion)
                pre_sess = sess   
        """ 마지막 세션 저장 """
        self.session_dataset += split(session)
            
        """ 추가 """
        # self.emoList = sorted(emoSet) # 항상 같은 레이블 순서를 유지하기 위해
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        f.close()
        
    def __len__(self): # 기본적인 구성
        return len(self.session_dataset)
    
    def __getitem__(self, idx): # 기본적인 구성
        return self.session_dataset[idx]
    
    def collate_fn(self, sessions): # 배치를 위한 구성
        '''
            input:
                data: [(session1), (session2), ... ]
            return:
                batch_input_tokens_pad: (B, L) padded
                batch_labels: (B)
        '''
        batch_input_token = []
        for session in sessions:
            input_token = ""
            for line in session:
                speaker, utt, emotion = line
                input_token += utt
            batch_input_token.append(input_token)
        
        return batch_input_token
      
dev_dataset = data_loader('./MELD/data/MELD/dev_sent_emo.csv')

""" 배치 결과 확인 """
from torch.utils.data import DataLoader
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dev_dataset.collate_fn)

i = 0
for data in dev_dataloader:
    print(i, data)
    i += 1
    if i > 2:
        break    

""" 배치 결과 확인 """
from torch.utils.data import DataLoader
dev_dataloader = DataLoader(dev_dataset, batch_size=3, shuffle=False, num_workers=4, collate_fn=dev_dataset.collate_fn)

for data in dev_dataloader:
    print(data[0])
    print(data[1])
    print(data[2])
    break

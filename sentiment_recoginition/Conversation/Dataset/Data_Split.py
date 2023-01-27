import glob
import csv

#데이터 연결
data_path = "./MELD/data/MELD/*.csv"
data_path_list = glob.glob(data_path)


# 데이터 출력
for data_path in data_path_list:
    f = open(data_path, 'r')
    rdr = csv.reader(f)
    
    for line in rdr:
        print(line)
        break
        
    f.close()
    break
    
# 세션으로 데이터 분할

"""
[발화1, 발화2, 발화3] --> 감정예측한다는 것은 발화3에 해당하는 감정을
여기서 발화1, 발화2에 해당하는 감정이 있고, 이를 학습데이터로 사용해야된다.
--> [발화1], [발화1, 발화2]
//
모델이 roberta: bidirectional
: attention이 양방향
[발화1, 발화2, 발화3]: 여기서 발화2에 해당하는 감정을 학습한다.
: 동시에 발화1에 해당하는 감정을 학습한다.
"""

# 데이터 저장
def split(session):
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])    
    return final_data
    
for data_path in data_path_list:
    f = open(data_path, 'r')
    rdr = csv.reader(f)
    
    """ 세션 데이터 저장할 것"""
    session_dataset = []
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
                # session_dataset.append(session)
                session_dataset += split(session)
                session = [[uniq_speaker, utt, emotion]]
                speaker_set = []
            pre_sess = sess   
    """ 마지막 세션 저장 """
    # session_dataset.append(session)
    session_dataset += split(session)
    f.close()
    
    """ 데이터 분할하기 """
    break

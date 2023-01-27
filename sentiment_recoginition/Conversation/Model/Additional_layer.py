""" score matrix """
clsNum = len(dev_dataset.emoList)
W = nn.Linear(hiddenDim, clsNum)
final_output = W(batch_com_final + batch_pm_gru_final)

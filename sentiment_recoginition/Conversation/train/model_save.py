import os
def SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    
import math
a = -(0.8*math.log(0.8)+0.2*math.log(0.2))
b = -(0.8*math.log(0.5)+0.2*math.log(0.5))
a, b

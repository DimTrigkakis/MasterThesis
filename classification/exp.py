import torch
import gc
from torch.autograd import Variable

import torch.utils.data as data

class CustomGarbageLeak(data.Dataset):

    def __init__(self):
        pass

    def __getitem__(self, index):
        t = torch.randn(50,50,50,50)
        #t = 5
        #t2 = torch.cat([t,t])
        return 0

    def __len__(self):
        return 10000
    
trainset = CustomGarbageLeak()
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=2, shuffle=True, num_workers=1)

if __name__ == "__main__":
    for i in range(3000):
        for j, datum in enumerate(trainloader):
            print(j)
            #del datum
            continue
            #print(datum,i,j)
            #datum = Variable(datum)

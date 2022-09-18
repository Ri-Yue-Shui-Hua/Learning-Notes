# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible
 
BATCH_SIZE = 5
 
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)
 
'''先转换成 torch 能识别的 Dataset'''
torch_dataset = Data.TensorDataset(x, y)
print(torch_dataset[0])     #输出(tensor(1.), tensor(10.))
print(torch_dataset[1])     #输出(tensor(2.), tensor(9.))
 
''' 把 dataset 放入 DataLoader'''
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    #num_workers=2,              # subprocesses for loading data
)
 
for epoch in range(3):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train your data...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
        
"""
Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [ 5.  4.  9.  8.  10.]
Epoch:  0 | Step:  1 | batch x:  [ 9.  10. 4.  8.  5.] | batch y:  [ 2.  1.  7.  3.  6.]
Epoch:  1 | Step:  0 | batch x:  [ 3.  4.  2.  9. 10.] | batch y:  [ 8.  7.  9.  2.  1.]
Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10. 4.  3.  6.  5.]
Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
Epoch:  2 | Step:  1 | batch x:  [10.  4.  8.  1.  5.] | batch y:  [ 1.  7.  3. 10.  6.]
"""
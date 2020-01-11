import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from gcommand_loader import GCommandLoader
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""# the network

# get the data to the code
"""

dataset_train = GCommandLoader('./data/train')
dataset_test = GCommandLoader('./data/test')

train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)


test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

class MyNet(torch.nn.Module):
  def __init__(self,image_size):
    super(MyNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
    self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2)
    self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=5, stride=1, padding=2)
    self.fc = nn.Linear(100,30)
    self.mp = nn.MaxPool2d(2)
    #self.dropout = nn.Dropout(p=0.5)
    
    self.loss_function = nn.CrossEntropyLoss()
   # self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    
    
  def forward(self,x):
    x = Variable(x)
    if torch.cuda.is_available():
      x = x.cuda()
    in_size = x.size(0)
    x = F.relu(self.mp(self.conv1(x)))
    x = F.relu(self.mp(self.conv2(x)))
    x = F.relu(self.mp(self.conv3(x)))
    
    # flat it
    x = x.veiw(in_size, -1)
    x = self.fc(x)
    
    
    return F.log_softmax(x)

model = MyNet(image_size=1*161*101)
for data, lebal in test_loader:
  model.forward(data)
  break
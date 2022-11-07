#%% Importing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import math
import time

from PoseDataset import PoseLandmarksDataset
from ToTensor import ToTensor
from Rotation import Rotation
from Scaling import Scaling
from Shearing import Shearing
from Brighness import Brightness
from Contrast import Contrast

#%%## Defining Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(48, 128, 5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.6)
        self.dropout2 = nn.Dropout2d(0.6)
        self.fc1 = nn.Linear(12800, 2048)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(2048, 28)
        self.lrn = nn.LocalResponseNorm(2)
        self.pool1 = nn.MaxPool2d(5,2,2)
        self.pool2 = nn.MaxPool2d(3,2,1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lrn(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.lrn(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
#%% Percent of Detected Joints (PDJ)
def PDJ(targets, outputs, fraction):
    # left hip: 4th joint ------ Right shoulder: 9th joint
    S = 0
    for i, target in enumerate(targets):
        torso_diameter = math.sqrt((target[6]-target[16])**2 + (target[7]-target[17])**2)
        output = outputs[i]
        for idx in range(0,outputs.size(1),2):
            if math.sqrt((target[idx]-output[idx])**2 + (target[idx+1]-output[idx+1])**2) <= fraction*torso_diameter:
                S += 1
#    return S/(targets.size(0)*targets.size(1)/2)
    return S
    
#%% Training Step
loss_train = []
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data = sample['image']
        target = sample['landmarks']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.float()
        output = model(data)
        loss = F.mse_loss(output, target.view_as(output).float())
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#%%## Testing Step
loss_valid = []
def valid(model, device, valid_loader, fraction):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in valid_loader:
            data = sample['image']
            target = sample['landmarks']
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            valid_loss += F.mse_loss(output, target.view_as(output).float(), reduction='sum').item()  # sum up batch loss
            pred = PDJ(target.view_as(output).float(), output, fraction)
            correct += pred
    valid_loss /= len(valid_loader.dataset)
    loss_valid.append(valid_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset)*14,
        100. * correct / (len(valid_loader.dataset)*14)))

#%% Main
def main():
    # Training settings
    fraction = 0.5
    batch_size = 64
    test_batch_size = 32
    epochs = 200
    lr = 0.0005
    no_coda = False
    seed = 1
    log_interval = 10
    save_model = True
    
    use_cuda = not no_coda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

#%% train dataset
    flag = 0
    # Original
    original = PoseLandmarksDataset('joints_train.csv',
                                    r'_images',
                                    transform=transforms.Compose([ToTensor()]))
    # Rotation
    rotated = PoseLandmarksDataset('joints_train.csv',
                                   r'_images',
                                   transform=transforms.Compose([Rotation(show = flag), ToTensor()]))
    
    # Scaling
    scaled = PoseLandmarksDataset('joints_train.csv',
                                  r'_images',
                                  transform=transforms.Compose([Scaling(show = flag), ToTensor()]))
    
    # Shearing
    sheared = PoseLandmarksDataset('joints_train.csv',
                                   r'_images',
                                   transform=transforms.Compose([Shearing(show = flag), ToTensor()]))
    
    # Brightness
    brightness_changed = PoseLandmarksDataset('joints_train.csv',
                                              r'_images',
                                              transform=transforms.Compose([Brightness(show = flag),
                                              ToTensor()]))
    
    # Contrast
    contrast_changed = PoseLandmarksDataset('joints_train.csv',
                                            r'_images',
                                            transform=transforms.Compose([Contrast(show = flag),
                                            ToTensor()]))
    
    # Concating datasets
    train_set = torch.utils.data.ConcatDataset((original, rotated, scaled,
                                                        sheared, brightness_changed, contrast_changed))

    #%%
    # valid dataset
    valid_set = PoseLandmarksDataset('joints_valid.csv', r'_images',
                                     transform=transforms.Compose([ToTensor()]))
    
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=batch_size, shuffle=True, **kwargs)

    # validation data loader
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=test_batch_size, shuffle=True, **kwargs)


    # make a network instance
#    PATH = 'lsp_1_200_saving_loss_F.pt'
    model = Net().to(device)
#    model.load_state_dict(torch.load(PATH))

    # configure optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    # configure learning rate scheduler
#    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train epochs
    time_start = time.time()
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        if not epoch%2:
            valid(model, device, valid_loader, fraction)
#        scheduler.step()
    time_end = time.time()
    print(time_end - time_start)

    # save the trained model
    if save_model:
	    torch.save(model.state_dict(), "lsp_1_200_Aug_F.pt")

#%% Calling main
if __name__ == '__main__':
    main()
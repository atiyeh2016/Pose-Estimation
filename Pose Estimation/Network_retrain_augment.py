#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PoseDataset import PoseLandmarksDataset
from ToTensor import ToTensor
import math
import time
import numpy as np
from Rotation_S import Rotation
#from Scaling import Scaling
#from Shearing import Shearing
#from Brighness import Brightness
#from Contrast import Contrast

fp = open('log_F.txt','w')

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

#%% Perventage of Correct Part
def PCP(targets, outputs):
    
    def ll (j1x,j2x,j1y,j2y): # Limb length
        return math.sqrt((target[j1x]-target[j2x])**2 + (target[j1y]-target[j2y])**2)
    S = 0
    for target, output in zip(targets, outputs):
        ALL = np.array([ll(target[0],target[2],target[1],target[3]),
                               ll(target[2],target[4],target[3],target[5]),
                               ll(target[4],target[6],target[5],target[7]),
                               ll(target[6],target[8],target[7],target[9]),
                               ll(target[8],target[10],target[9],target[11]),
                               ll(target[12],target[14],target[13],target[15]),
                               ll(target[14],target[16],target[15],target[17]),
                               ll(target[16],target[18],target[17],target[19]),
                               ll(target[18],target[20],target[19],target[21]),
                               ll(target[20],target[22],target[21],target[23]),
                               ll(target[24],target[26],target[25],target[27])]) # Actual

        PLL = np.array([ll(output[0],output[2],output[1],output[3]),
                               ll(output[2],output[4],output[3],output[5]),
                               ll(output[4],output[6],output[5],output[7]),
                               ll(output[6],output[8],output[7],output[9]),
                               ll(output[8],output[10],output[9],output[11]),
                               ll(output[12],output[14],output[13],output[15]),
                               ll(output[14],output[16],output[15],output[17]),
                               ll(output[16],output[18],output[17],output[19]),
                               ll(output[18],output[20],output[19],output[21]),
                               ll(output[20],output[22],output[21],output[23]),
                               ll(output[24],output[26],output[25],output[27])]) # Predicted
        S += np.sum(PLL>0.5*ALL)
        
    return S
    
#%% Training Step
loss_train = []
def train(args, model, device, train_loader, optimizer, epoch):
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
        if batch_idx % args.log_interval == 0:
            fp.writelines('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#%%## Testing Step
loss_valid = []
def valid(args, model, device, valid_loader, fraction):
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
#            pred = PDJ(target.view_as(output).float(), output, fraction)
            correct += pred
    valid_loss /= len(valid_loader.dataset)
    loss_valid.append(valid_loss)
    
    fp.writelines('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset)*14,
        100. * correct / (len(valid_loader.dataset)*14)))
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset)*14,
        100. * correct / (len(valid_loader.dataset)*14)))

#%% Main
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LSP')
    parser.add_argument('--fraction', type=int, default=0.5, metavar='N',
                        help='fraction of torso diameter (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

#%% train dataset
    flag = 0
    # Original
    original = PoseLandmarksDataset('joints_train.csv',
                                    r'_images',
                                    transform=transforms.Compose([ToTensor()]))
    # Rotation
    rotated = PoseLandmarksDataset('joints_rotate_train.csv',
                                   r'images',
                                   transform=transforms.Compose([Rotation(show = flag), ToTensor()]))
    
    # Scaling
#    scaled = PoseLandmarksDataset('joints_train.csv',
#                                  r'_images',
#                                  transform=transforms.Compose([Scaling(show = flag), ToTensor()]))
#    
#    # Shearing
#    sheared = PoseLandmarksDataset('joints_train.csv',
#                                   r'_images',
#                                   transform=transforms.Compose([Shearing(show = flag), ToTensor()]))
#    
#    # Brightness
#    brightness_changed = PoseLandmarksDataset('joints_train.csv',
#                                              r'_images',
#                                              transform=transforms.Compose([Brightness(show = flag),
#                                              ToTensor()]))
#    
#    # Contrast
#    contrast_changed = PoseLandmarksDataset('joints_train.csv',
#                                            r'_images',
#                                            transform=transforms.Compose([Contrast(show = flag),
#                                            ToTensor()]))
    
    # Concating datasets
    train_set = torch.utils.data.ConcatDataset((original, rotated))
                                                        
    #%%
    # valid dataset
    valid_set = PoseLandmarksDataset('joints_train.csv', r'_images',
                                     transform=transforms.Compose([ToTensor()]))
    
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    # validation data loader
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # make a network instance
    PATH = 'lsp_1_200_Aug_F.pt'
    model = Net().to(device)
    model.load_state_dict(torch.load(PATH))

    # configure optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # configure learning rate scheduler
#    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train epochs
    time_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
#        if not epoch%2:
        valid(args, model, device, valid_loader, args.fraction)
#        scheduler.step()
    time_end = time.time()
    print(time_end - time_start)

    # save the trained model
    if args.save_model:
	    torch.save(model.state_dict(), "lsp_201_400_Aug_F.pt")

    # run inference
#    if args.save_model:
#	    model = Net()
#	    model.load_state_dict(torch.load("lsp_cnn.pt"))
#	    sample = next(iter(valid_loader))
#	    model.eval()
#	    outputs = model(sample['image'].float())
#	    _, lbl = torch.max(outputs.data, 1)
#	    print('\nThe true lable: ', sample['landmarks'][0].item())
#	    print('The classifier lable: ', lbl[0].item())

#%% Calling main
if __name__ == '__main__':
    main()

fp.close()
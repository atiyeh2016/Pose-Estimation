#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PoseDataset import PoseLandmarksDataset
from ToTensor import ToTensor
import math
import time
import numpy as np

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
PDJ_All = np.zeros([14,400])
def PDJ(targets, outputs, fraction, epoch):
    # left hip: 4th joint ------ Right shoulder: 9th joint
    S = 0
    for i, target in enumerate(targets):
        torso_diameter = math.sqrt((target[6]-target[16])**2 + (target[7]-target[17])**2)
        output = outputs[i]
        for idx in range(0,outputs.size(1),2):
            pdj_idx = math.sqrt((target[idx]-output[idx])**2 + (target[idx+1]-output[idx+1])**2)
            PDJ_All[int(idx/2),epoch-1] = pdj_idx
            if pdj_idx <= fraction*torso_diameter:
                S += 1
#    return S/(targets.size(0)*targets.size(1)/2)
    return S

#%% Perventage of Correct Part
PCP_All = np.zeros([11,400])
def PCP(targets, outputs, epoch):
    
    def ll (j1x,j2x,j1y,j2y): # Limb length
        return math.sqrt((j1x-j2x)**2 + (j1y-j2y)**2)
    S = 0
    for target, output in zip(targets, outputs):
        ALL = np.array([ll(target[0].item(),target[2].item(),target[1].item(),target[3].item()),
                               ll(target[2].item(),target[4].item(),target[3].item(),target[5].item()),
                               ll(target[4].item(),target[6].item(),target[5].item(),target[7].item()),
                               ll(target[6].item(),target[8].item(),target[7].item(),target[9].item()),
                               ll(target[8].item(),target[10].item(),target[9].item(),target[11].item()),
                               ll(target[12].item(),target[14].item(),target[13].item(),target[15].item()),
                               ll(target[14].item(),target[16].item(),target[15].item(),target[17].item()),
                               ll(target[16].item(),target[18].item(),target[17].item(),target[19].item()),
                               ll(target[18].item(),target[20].item(),target[19].item(),target[21].item()),
                               ll(target[20].item(),target[22].item(),target[21].item(),target[23].item()),
                               ll(target[24].item(),target[26].item(),target[25].item(),target[27].item())]) # Actual

        PLL = np.array([ll(output[0].item(),output[2].item(),output[1].item(),output[3].item()),
                               ll(output[2].item(),output[4].item(),output[3].item(),output[5].item()),
                               ll(output[4].item(),output[6].item(),output[5].item(),output[7].item()),
                               ll(output[6].item(),output[8].item(),output[7].item(),output[9].item()),
                               ll(output[8].item(),output[10].item(),output[9].item(),output[11].item()),
                               ll(output[12].item(),output[14].item(),output[13].item(),output[15].item()),
                               ll(output[14].item(),output[16].item(),output[15].item(),output[17].item()),
                               ll(output[16].item(),output[18].item(),output[17].item(),output[19].item()),
                               ll(output[18].item(),output[20].item(),output[19].item(),output[21].item()),
                               ll(output[20].item(),output[22].item(),output[21].item(),output[23].item()),
                               ll(output[24].item(),output[26].item(),output[25].item(),output[27].item())]) # Predicted
        S += np.sum(PLL<0.5*ALL)
        PCP_All[:,epoch] = (PLL>0.5*ALL).astype(int)
        
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
def valid(args, model, device, valid_loader, fraction, epoch):
    model.eval()
    valid_loss = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for sample in valid_loader:
            data = sample['image']
            target = sample['landmarks']
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            valid_loss += F.mse_loss(output, target.view_as(output).float(), reduction='sum').item()  # sum up batch loss
            pred1 = PDJ(target.view_as(output).float(), output, fraction, epoch)
            pred2 = PCP(target.view_as(output).float(), output, epoch)
            correct1 += pred1
            correct2 += pred2
    valid_loss /= len(valid_loader.dataset)
    loss_valid.append(valid_loss)
    
    fp.writelines('\nTest set: Average loss: {:.4f}, PCJ: {}/{} ({:.0f}%) \
    PCP: {}/{} ({:.0f}%)\n'.format(valid_loss, correct1, len(valid_loader.dataset)*14,
        100. * correct1 / (len(valid_loader.dataset)*14), correct2, len(valid_loader.dataset)*14,
        100. * correct2 / (len(valid_loader.dataset)*14)))
    
    print('\nTest set: Average loss: {:.4f}, PCJ: {}/{} ({:.0f}%) \
    PCP: {}/{} ({:.0f}%)\n'.format(valid_loss, correct1, len(valid_loader.dataset)*14,
        100. * correct1 / (len(valid_loader.dataset)*14), correct2, len(valid_loader.dataset)*14,
        100. * correct2 / (len(valid_loader.dataset)*14)))

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

    # train dataset
    train_set = PoseLandmarksDataset('joints_train.csv', r'_images',
                                     transform=transforms.Compose([ToTensor()]))
    
    # valid dataset
    valid_set = PoseLandmarksDataset('joints_valid.csv', r'_images',
                                     transform=transforms.Compose([ToTensor()]))
    
    # test dataset
    test_set = PoseLandmarksDataset('joints_test.csv', r'_images',
                                    transform=transforms.Compose([ToTensor()]))

    # train data loader
    train_loader = torch.utils.data.DataLoader(train_set, 
    	batch_size=args.batch_size, shuffle=True, **kwargs)

    # validation data loader
    valid_loader = torch.utils.data.DataLoader(valid_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # test data loader
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # make a network instance
#    PATH = 'lsp_1_200_saving_loss_F.pt'
    model = Net().to(device)
#    model.load_state_dict(torch.load(PATH))

    # configure optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # configure learning rate scheduler
#    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train epochs
    time_start = time.time()
    for epoch in range(1, 400 + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        valid(args, model, device, train_loader, args.fraction, epoch)
#        scheduler.step()
    time_end = time.time()
    print(time_end - time_start)

    # save the trained model
    if args.save_model:
	    torch.save(model.state_dict(), "lsp_1_400_saving_loss_S_PCP_PCJ.pt")

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
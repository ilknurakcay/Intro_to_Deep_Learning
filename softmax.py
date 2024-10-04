import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


# Hyperparameters
random_seed = 123
learning_rate = 0.1
num_epochs = 4
batch_size = 256

num_features = 784
num_classes = 10




train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)


# Dataset check
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape) 
    print('Image label dimensions:', labels.shape)
    break

class SoftmaxReg(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super(SoftmaxReg,self).__init__()
        self.linear=torch.nn.Linear(num_features,num_classes)
        #initial weight and bias equal to zero
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
    def forward(self,x):
        logits=self.linear(x)
        probas=F.softmax(logits,dim=1)
        return logits,probas
model = SoftmaxReg(num_features=num_features,
                          num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
torch.manual_seed(random_seed)
def compute_acc(model,DataLoader):
    correct_pred,num_examples=0,0
    for features,targets in DataLoader:
        breakpoint()
        #flatten (8x28 to 784)
        features=features.view(-1,28*28)
        targets=targets
        #Logits are value that before apply  softmax
        #Probas , probability that its image belongs to a class ex: For 1, 0.8 and for 2, 0.2
        logits, probas = model(features)
        #Returns the highest probability in the array and the index where this probability is found.
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100
    

start_time = time.time()
epoch_costs = []
for epoch in range(num_epochs):
    avg_cost = 0.
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.view(-1, 28*28)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        

        cost = F.cross_entropy(logits, targets)
        #Resets gradients before backpropagation. 
        optimizer.zero_grad()
        cost.backward()
        avg_cost += cost
        
        optimizer.step()
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_dataset)//batch_size, cost))
            
    with torch.set_grad_enabled(False):
        avg_cost = avg_cost/len(train_dataset)
        epoch_costs.append(avg_cost)
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_acc(model, train_loader)))
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    
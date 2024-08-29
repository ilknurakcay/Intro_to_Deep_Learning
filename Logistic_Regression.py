import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
### DATASET

data = np.genfromtxt('.//toydata.txt', delimiter='\t')
print(len(data))
x = data[:, :2].astype(np.float32)
y = data[:, 2].astype(np.int64)

np.random.seed(123)
idx = np.arange(y.shape[0])
print("y.shape[0]",y.shape[0])
np.random.shuffle(idx)
print("idx",idx)
print("idx[:25]",idx[:25])
X_test, y_test = x[idx[:25]], y[idx[:25]]
X_train, y_train = x[idx[25:]], y[idx[25:]]
print("x train",len(X_train))
print("x test",len(X_test))

mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])

plt.xlim([x[:, 0].min()-0.5, x[:, 0].max()+0.5])
plt.ylim([x[:, 1].min()-0.5, x[:, 1].max()+0.5])
plt.show()

class LogisticRegression1():
    def __init__(self,num_features):
        self.num_features=num_features
        self.weights=torch.zeros(1,num_features,dtype=torch.float32) #num features = num weight 
        self.bias=torch.zeros(1,dtype=torch.float32)
    
    def forward(self,x):
        """
        y=xw+b
        """
        linear = torch.add(torch.mm(x, self.weights.t()), self.bias).view(-1) # net input
        probas=self._sigmoid(linear) #  
        return probas
    def backward(self,x,y,probas):
        """
        gradient descent computing
        dL/dw=dL/da * da/dz* dz/dw =>dL/dw
        """
        grad_loss_wrt_z=probas.view(-1)-y #gradient loss with respect to output => dL/dz = a-y
        grad_loss_wrt_w=torch.mm(x.t(),grad_loss_wrt_z.view(-1,1)).t() #dL/dw = (a-y)*x
        grad_loss_wrt_b=torch.sum(grad_loss_wrt_z)
        return grad_loss_wrt_w,grad_loss_wrt_b
    
    def predict_labels(self,x):
        """
        treshold function
        """
        probas=self.forward(x)
        labels=torch.where(probas >= .5,1,0)#if prob > 0,5, class= 1,  else 0
        return labels

    def evaluate(self,x,y):
        labels=self.predict_labels(x).float()
        accuracy=torch.sum(labels.view(-1)==y.float()).item()/y.size(0)
        return accuracy


    def _sigmoid(self,z):
        return 1./(1. + torch.exp(-z))
    
    def _logit_cost(self,y,proba):
        """
        logistic cost function
        """
        tmp1=torch.mm(-y.view(1,-1),torch.log(proba.view(-1,1)))
        tmp2=torch.mm(((1-y)).view(1,-1),torch.log(1- proba.view(-1,1)))
        return tmp1-tmp2

    def train(self, x, y, num_epochs, learning_rate=0.01):
        epoch_cost = []
        for e in range(num_epochs):
                
                #### Compute outputs : y=x*w+b
            probas = self.forward(x)
                
                #### Compute gradients 
            grad_w, grad_b = self.backward(x, y, probas)

                #### Update weights
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
                
                #### Logging ####
            cost = self._logit_cost(y, self.forward(x)) / x.size(0)
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % cost)
            epoch_cost.append(cost)
        return epoch_cost



X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

model1 = LogisticRegression1(num_features=2)
epoch_cost = model1.train(X_train_tensor, y_train_tensor, num_epochs=30, learning_rate=0.1)



print('\nModel parameters:')
print('  Weights: %s' % model1.weights)
print('  Bias: %s' % model1.bias)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

test_acc = model1.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))

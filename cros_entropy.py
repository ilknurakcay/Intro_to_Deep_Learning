import torch
def to_onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot

y = torch.tensor([0, 1, 2, 2])
y_enc = to_onehot(y, 3)
print('one-hot encoding:\n', y_enc)

#I defined random Z
Z = torch.tensor( [[-0.3,  -0.5, -0.5],
                   [-0.4,  -0.1, -0.5],
                   [-0.3,  -0.94, -0.5],
                   [-0.99, -0.88, -0.5]])

def softmax(z):
    """
    use for multiclass clasification.Each value in the vector transform  to probabilty.
    """
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

smax = softmax(Z)
print('softmax:\n', smax)
"""
 tensor([[0.3792, 0.3104, 0.3104],
        [0.3072, 0.4147, 0.2780],
        [0.4263, 0.2248, 0.3490],
        [0.2668, 0.2978, 0.4354]])
"""

def to_classlabel(z):
    """
    The value with the highest probability is converted to class
    For ex:[0.3792, 0.3104, 0.3104] --> class 0
    """
    return torch.argmax(z, dim=1)

print('predicted class labels: ', to_classlabel(smax))
print('true class labels: ', to_classlabel(y_enc))
"""
predicted class labels:  tensor([0, 1, 0, 2])
true class labels:  tensor([0, 1, 2, 2])
"""

def cross_entropy(softmax, y_target):
    """
The lower the entropy values, the better the performance of the model.
    """
    return - torch.sum(torch.log(softmax) * (y_target), dim=1)

xent = cross_entropy(smax, y_enc)

#Cross Entropy: tensor([0.9698, 0.8801, 1.0527, 0.8314]), 3rd prediction is high, prediction is not true
print('Cross Entropy:', xent)

#In PyTorch
import torch.nn.functional as F
F.nll_loss(torch.log(smax), y, reduction='none') #if not use reduction='none' , function return average value
F.cross_entropy(Z, y, reduction='none')

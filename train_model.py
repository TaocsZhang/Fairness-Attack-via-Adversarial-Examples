# Training neural net
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import keras

class GermanNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(GermanNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear2(h2))
        h4 = self.relu(self.linear2(h3))
        h5 = self.relu(self.linear2(h4))
        h6 = self.relu(self.linear2(h5))
        a3 = self.linear3(h6)
        y = self.softmax(a3)
        return y

def set_grad(var):
    '''function to extract grad
    '''

    def hook(grad):
        var.grad = grad
    return hook

def train(model, criterion, optimizer, config):
    '''training model
    '''
    model.train()
    df_train = config['TrainData']
    target = config['Target']
    feature_names = config['FeatureNames']
    X = torch.FloatTensor(df_train[feature_names].values)
    y = keras.utils.to_categorical(df_train[target], 2)
    y = torch.FloatTensor(y)
    N = 100 # batch size
    current_loss = 0
    current_correct = 0

    # Training in batches
    for ind in range(0, X.size(0), N):
        indices = range(ind, min(ind + N, X.size(0)) - 1)
        inputs, labels = X[indices], y[indices]
        inputs = Variable(inputs, requires_grad=True)

        optimizer.zero_grad()
        output = model(inputs)

        _, indices = torch.max(output, 1)
        preds = torch.tensor(keras.utils.to_categorical(indices, 2))
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        current_correct += (preds.int() == labels.int()).sum() / 2

    current_loss = current_loss / X.size(0)
    current_correct = current_correct.double() / X.size(0)
    return preds, current_loss, current_correct.item()

def test(model, criterion, config):
    '''

    :param model:
    :return:

    Note the difference between model.eval() and torch.no_grad()
    model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).

    '''
    model.eval()
    df_test = config['TestData']
    feature_names = config['FeatureNames']
    target = config['Target']
    X = torch.FloatTensor(df_test[feature_names].values)
    y = keras.utils.to_categorical(df_test[target], 2)
    y = torch.FloatTensor(y)
    N = 100 # batch size

    current_loss = 0
    current_correct = 0
    softmax_output_grad_list = []
    softmax_output_list = []
    pred_list = []
    # Test in batches
    # with torch.no_grad():
    for ind in range(0, X.size(0), N):
        indices = range(ind, min(ind + N, X.size(0)))
        inputs, labels = X[indices], y[indices]
        inputs = Variable(inputs, requires_grad=True)

        output = model(inputs)
        output.register_hook(set_grad(output))

        _, indices = torch.max(output, 1)
        preds = torch.tensor(keras.utils.to_categorical(indices, 2))
        y_pred = np.argmax(preds.detach().numpy(), axis=1)
        loss = criterion(output, labels)
        loss.backward()

        pred_list.append(y_pred)
        softmax_output_grad_list.append(output.grad.tolist())
        softmax_output_list.append(output.tolist())
        current_loss += loss.item()
        current_correct += (preds.int() == labels.int()).sum() / 2

    current_loss = current_loss / X.size(0)
    current_correct = current_correct.double() / X.size(0)
    return pred_list, current_loss, current_correct.item(), softmax_output_list, softmax_output_grad_list

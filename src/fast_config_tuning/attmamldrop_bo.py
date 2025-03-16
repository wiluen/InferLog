import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

Attention_in_features=3  
Attention_hidden_features=32
class AttentionMechanism(nn.Module):
    def __init__(self):
        super(AttentionMechanism,self).__init__()

        self.Qweight = nn.Parameter(torch.rand(Attention_in_features, Attention_hidden_features) *
                                    ((4 / Attention_in_features) ** 0.5) -
                                    (1 / Attention_in_features) ** 0.5)
        self.Kweight = nn.Parameter(torch.rand(Attention_in_features, Attention_hidden_features) *
                                    ((4 / Attention_in_features) ** 0.5) -
                                    (1 / Attention_in_features) ** 0.5)

    def forward(self, s):
        current_task = s[0:1, :]  
        existing_tasks = s[1:, :]
        q = torch.matmul(current_task, self.Qweight) 
        k = torch.matmul(existing_tasks, self.Kweight)  
        att = torch.matmul(q, k.transpose(-1, -2)) 
        att = att / (torch.norm(att, p=2, dim=-1, keepdim=True) + 0.001) 
        att = torch.softmax(att, dim=-1)
        return att

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        weight1 = nn.Parameter(torch.ones([64, 3]))
        bias1 = nn.Parameter(torch.zeros(64))
        init.kaiming_normal_(weight1, mode='fan_out', nonlinearity='relu')
        init.zeros_(bias1)  
        self.vars.extend([weight1, bias1])
        #bn1
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        # fc2
        weight2 = nn.Parameter(torch.ones([1, 64]))
        bias2 = nn.Parameter(torch.zeros(1))
        init.kaiming_normal_(weight2, mode='fan_out', nonlinearity='relu')
        init.zeros_(bias2) 
        self.vars.extend([weight2, bias2])

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, params=None):
        '''
        Define the forward propagation of the model
        : param x:  input data
        : param params:  The externally transmitted parameter list defaults to None
        : return:  Output of the model
        '''
        if params is None:
            params = self.vars

        weight, bias = params[0], params[1]  
                      
        x = F.linear(x, weight, bias) 
        weight_bn, bias_bn = params[2], params[3]
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight_bn, bias=bias_bn, training=True)
        x = F.relu(x) 
        if self.enable_dropout:  # MC dropout
            x = self.dropout(x)
        weight, bias = params[4], params[5]
        x = F.linear(x, weight, bias)
        return x

    def parameters(self):
        return self.vars

class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        super(MetaLearner, self).__init__()
        self.update_step = 5
        self.update_step_test = 5
        self.net = BaseNet() 
        self.meta_lr = 1e-3
        self.base_lr = 1e-4
        self.att_lr=1e-4 
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr) 
        self.attention=AttentionMechanism()
        self.attention_optim=torch.optim.Adam(self.attention.parameters(),lr=self.att_lr)

    def finetune(self, x, y,fast_weights):
        self.net.enable_dropout=False
        for k in range(5):
            print('meta model finetune step',k)
            y_hat = self.net(x, params=fast_weights)
            loss = F.mse_loss(y_hat, y.unsqueeze(-1))
            grad = torch.autograd.grad(loss, fast_weights,allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))
        return fast_weights
    
    def predict(self,x,fast_weights):
        self.net.enable_dropout=True
        outputs = []
        for _ in range(10):
            output = self.net(x, params=fast_weights)
            outputs.append(output.detach().numpy())
        outputs = np.array(outputs)
        mean_output = np.mean(outputs, axis=0)[:,0]
        std_output = np.std(outputs, axis=0)[:,0]
        print(mean_output,std_output)
        return mean_output,std_output

            




import torch
import numpy as np
import os
import time
import pandas as pd
import random
import scipy.stats as stats
from torch import nn
from torch.nn import functional as F
from copy import deepcopy, copy
import torch.nn.init as init
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

shot=10

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
        self.scale = torch.sqrt(torch.tensor(Attention_hidden_features, dtype=torch.float32))

    def forward(self, s):
        current_task = s[0:1, :]
        existing_tasks = s[1:, :]
        q = torch.matmul(current_task, self.Qweight) 
        k = torch.matmul(existing_tasks, self.Kweight) 
        att = torch.matmul(q, k.transpose(-1, -2))  # 点积
        att = att / (torch.norm(att, p=2, dim=-1, keepdim=True) + 0.001) 
        att = torch.softmax(att, dim=-1)
        return att

def get_similar(task):
    df = pd.read_csv('log_feature.csv')
    data_dict = {row['dataset']: np.array([row['feature1'],row['feature2'],row['feature3']]) for index, row in df.iterrows()}
    feature = None
    if task in data_dict:
        feature = data_dict[task]
    else:
        print(f"No data found for task: {task}")
    spark = data_dict.get('Spark', None)
    mac = data_dict.get('Mac', None)
    hdfs = data_dict.get('HDFS', None)
    bgl = data_dict.get('BGL', None)
    hadoop = data_dict.get('Hadoop', None)
    zookeeper = data_dict.get('Zookeeper', None)
    openstack = data_dict.get('OpenStack', None)
    proxifier = data_dict.get('Proxifier', None)
    apache = data_dict.get('Apache', None)
    healthapp = data_dict.get('HealthApp', None)
    linux = data_dict.get('Linux', None)
    openssh = data_dict.get('OpenSSH', None)
    windows = data_dict.get('Windows', None)
    thunderbird = data_dict.get('Thunderbird', None)
    android = data_dict.get('Android', None)
    hpc = data_dict.get('HPC', None)

    if task=='Spark':feature=spark
    if task=='Mac':feature=mac
    if task=='HDFS':feature=hdfs
    if task=='BGL':feature=bgl
    if task=='Hadoop':feature=hadoop
    if task=='Zookeeper':feature=zookeeper
    if task=='OpenStack':feature=openstack
    if task=='Proxifier':feature=proxifier
    if task=='Apache':feature=apache
    if task=='HealthApp':feature=healthapp
    if task=='Linux':feature=linux
    if task=='OpenSSH':feature=openssh
    if task=='Windows':feature=windows
    if task=='Thunderbird':feature=thunderbird
    if task=='Android':feature=android
    if task=='HPC':feature=hpc

    tasks_features = np.array([feature,spark, mac, hdfs,bgl,hadoop,openstack,zookeeper,proxifier])  #meta-train #1
    tasks_features_batch = torch.tensor(tasks_features, dtype=torch.float32) 
    tasks_features_batch = tasks_features_batch
    return tasks_features_batch

def data_loader(mode,task,shot):
    if mode=='train':
        datasets=['apache','healthapp','linux','openssh','windows','thunderbird','android','hpc']
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        for dataset in datasets:
            df = pd.read_csv(f'dataset/{dataset}.csv')
            X = df[['conf1', 'conf2', 'conf3']].values
            y = df['p95latency'].values
            indices = list(range(len(df))) 
            random.shuffle(indices)  
            support_indices = indices[:15] 
            query_indices = indices[15:60] 

            support_x.append(X[support_indices])
            support_y.append(y[support_indices]) 

            query_x.append(X[query_indices])
            query_y.append(y[query_indices])

        support_x = np.array(support_x) 
        support_y = np.array(support_y)

        query_x = np.array(query_x)
        query_y = np.array(query_y)

    elif mode=='test': 
        df = pd.read_csv(f'dataset/{task}.csv')
        X = df[['conf1', 'conf2', 'conf3']].values
        y = df['p95latency'].values
        indices = list(range(len(df))) 
        if shot==10:
            support_indices = indices[:10] 
            query_indices = indices[10:40] 
        elif shot==5:
            support_indices = indices[:5] 
            query_indices = indices[10:25]
        elif shot==15:
            support_indices = indices[:15] 
            query_indices = indices[10:55]
        support_x=X[support_indices]  
        support_y=y[support_indices]  

        query_x=X[query_indices] 
        query_y=y[query_indices]

        support_x = np.array(support_x)
        support_y = np.array(support_y)

        query_x = np.array(query_x)  
        query_y = np.array(query_y)
    return support_x, support_y, query_x, query_y



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

        weight, bias = params[4], params[5]
        x = F.linear(x, weight, bias) 

        return x

    def parameters(self):
        return self.vars




class MetaLearner(nn.Module):
    def __init__(self):
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

    def forward(self, x_spt, y_spt, x_qry, y_qry,task):
        # 初始化
        task_num, sample_num,feature = x_spt.size()  
        query_size = x_qry.size(1) 
        loss_list_qry = [[] for _ in range(self.update_step+1)]
        error_list = [0 for _ in range(self.update_step+1)]

        for i in range(task_num): 
            y_hat = self.net(x_spt[i], params=None) 
            loss = F.mse_loss(y_hat, y_spt[i].unsqueeze(-1))
            grad = torch.autograd.grad(loss, self.net.parameters())
            tuples = zip(grad, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples)) 

            with torch.no_grad():
                y_hat = self.net(x_qry[i], list(self.net.parameters()))
                loss_qry = F.mse_loss(y_hat, y_qry[i].unsqueeze(-1))
                loss_list_qry[0].append(loss_qry)
                pred_qry = torch.abs(y_hat - y_qry[i].unsqueeze(-1)) / torch.abs(y_qry[i].unsqueeze(-1))
                error_list[0] += pred_qry
            for k in range(1, self.update_step):
                y_hat = self.net(x_spt[i], params=fast_weights) 
                loss = F.mse_loss(y_hat, y_spt[i].unsqueeze(-1))
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
                y_hat = self.net(x_qry[i], params=fast_weights)
                loss_qry = F.mse_loss(y_hat, y_qry[i].unsqueeze(-1))
                loss_list_qry[k+1].append(loss_qry)
                with torch.no_grad():
                    pred_qry = torch.abs(y_hat - y_qry[i].unsqueeze(-1)) / torch.abs(y_qry[i].unsqueeze(-1))
                    error_list[k+1] += pred_qry

        tasks_features_batch=get_similar(task)  
        similar_weight=self.attention(tasks_features_batch)          
        weighted_loss_sum = torch.sum(torch.stack(loss_list_qry[-1]) * similar_weight)
        weights_sum = torch.sum(similar_weight)
        loss_qry = weighted_loss_sum / weights_sum

        self.meta_optim.zero_grad()
        self.attention_optim.zero_grad()
        loss_qry.backward()
        self.meta_optim.step()
        self.attention_optim.step()

        accs = error_list[-1]/task_num
        loss = [float(t.detach().cpu()) for t in loss_list_qry[-1]]
        training_loss=np.sum(loss)/task_num
        return accs, training_loss,y_hat,y_qry[task_num-1],similar_weight

    def test(self, x_spt, y_spt, x_qry, y_qry):
        error_list = [0 for _ in range(self.update_step_test + 1)]
        new_net = deepcopy(self.net)
        y_hat = new_net(x_spt)
        loss = F.mse_loss(y_hat, y_spt.unsqueeze(-1))
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))
        with torch.no_grad():
            y_hat = new_net(x_qry, params=new_net.parameters())
            pred_qry = torch.abs(y_hat - y_qry.unsqueeze(-1)) / torch.abs(y_qry.unsqueeze(-1))
            error_list[0] += pred_qry
        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params=fast_weights)
            loss = F.mse_loss(y_hat, y_spt.unsqueeze(-1))
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

            y_hat = new_net(x_qry, fast_weights)
            with torch.no_grad():
                pred_qry = torch.abs(y_hat - y_qry.unsqueeze(-1)) / torch.abs(y_qry.unsqueeze(-1))    
                error_list[k+1] += pred_qry
        del new_net
        return loss,error_list[-1],y_hat,y_qry
    
    def finetune(self,x_spt, y_spt, x_qry,task):
        state_dict = torch.load(f'model/attmaml_{task}.pth')
        fast_weights = list(state_dict.values())
        for param in fast_weights:
            param.requires_grad = True
        for k in range(10):
            y_hat = self.net(x_spt, params=fast_weights)
            loss = F.mse_loss(y_hat, y_spt.unsqueeze(-1))
            grad = torch.autograd.grad(loss, fast_weights,allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad[:6], fast_weights[:6])))
            with torch.no_grad():
                error = torch.abs(y_hat - y_spt.unsqueeze(-1)) / torch.abs(y_spt.unsqueeze(-1))    
            print(torch.mean(error))
        print('10-shot finetine complete')
        y_pred = self.net(x_qry, params=fast_weights)
        return error,y_pred
            

            
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = MetaLearner()
    epochs =5000 
    shot=5
    min_error=0.1
    task='Spark'
    train=False
    if train==True:
        for step in range(epochs):
            start = time.time() 
            x_spt, y_spt, x_qry, y_qry = data_loader('train',task,shot)

            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float(), torch.from_numpy(y_spt).float(), torch.from_numpy(x_qry).float(), torch.from_numpy(y_qry).float()


            accs, loss,y_hat,x_qry = meta(x_spt, y_spt, x_qry, y_qry,task) 
            end = time.time()

            if step % 100 == 1: 
                print(f"Epoch: {step}, Time: {end - start:.2f}s")
                print(f"Training Loss: {loss}")
                print(f"relative error: {accs}")

            if step % 100 == 1 and step>100: 
                accs = []
                x_spt, y_spt, x_qry, y_qry = data_loader('test',task,shot) 
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float(), torch.from_numpy(y_spt).float(), torch.from_numpy(x_qry).float(), torch.from_numpy(y_qry).float()
                accs,y_hat,x_qry = meta.test(x_spt, y_spt, x_qry, y_qry)

                for pred, actual in zip(y_hat, x_qry):
                    print(f"predict:{pred},gt:{actual}")
                print(f"relative error: {accs}")
                if float(torch.mean(accs))<min_error:
                    min_error=float(torch.mean(accs))
                    print('min error',min_error)
                    params = [param.data for param in meta.net.parameters()]
                    torch.save(meta.net.state_dict(), f'model/maml_{task}.pth')
                print('min error',min_error)

    else:
        mse_list=[]
        mae_list=[]
        mape_list=[]
        for i in range(10): 
            x_spt, y_spt, x_qry, y_qry = data_loader('test',task,shot)
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float(), torch.from_numpy(y_spt).float(), torch.from_numpy(x_qry).float(), torch.from_numpy(y_qry).float()
            accs,y_pred = meta.finetune(x_spt, y_spt, x_qry,task) 
            y_qry=np.array(y_qry)
            y_pred=y_pred.detach().numpy()
            for pred, actual in zip(y_pred, y_qry):
                print(f"predict:{pred},gt:{actual}")
            mse = mean_squared_error(y_qry, y_pred)
            mae = mean_absolute_error(y_qry, y_pred)
            mape=mean_absolute_percentage_error(y_qry,y_pred)
            mse_list.append(mse)
            mape_list.append(mape)
            mae_list.append(mae)
        print(mse_list,mae_list,mape_list)
        print('mse:',np.mean(mse_list))
        print('mae:',np.mean(mae_list))
        print('mape:',np.mean(mape_list))


if __name__=='__main__':
    main()


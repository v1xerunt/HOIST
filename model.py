
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.nn.utils import weight_norm

RANDOM_SEED = 123
def seed_torch():
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HOIST_without_claim(nn.Module):
    def __init__(self, dynamic_dims, static_dims, rnn_dim, device):
        super(HOIST_without_claim, self).__init__()
        self.dynamic_dims = dynamic_dims
        self.static_dims = static_dims
        self.device = device
        self.rnn_dim = rnn_dim
        
        self.covid_weight = nn.Sequential(nn.Linear(1, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 1), nn.Sigmoid())
        #self.claim_weight = nn.Sequential(nn.Linear(20, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 20), nn.Sigmoid())
        self.hos_weight = nn.Sequential(nn.Linear(4, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 4), nn.Sigmoid())
        #self.vac_weight = nn.Sequential(nn.Linear(17, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 17), nn.Sigmoid())
        
        self.rnn = nn.LSTM(dynamic_dims, rnn_dim, batch_first=True)
        
        self.linear = nn.Linear(rnn_dim, rnn_dim)
        self.linear_2 = nn.Linear(rnn_dim, 1)
        
        pop_dim, demo_dim, eco_dim = static_dims
        self.W_pop = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(pop_dim, pop_dim).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_pop = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*pop_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.W_demo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(demo_dim, demo_dim).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_demo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*demo_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.W_eco = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(eco_dim, eco_dim).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_eco = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*eco_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.W_geo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2, 2).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_geo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
    
    def forward(self, dynamic, static, h0=None):
        pop, demo, eco, geo = static
        N = pop.shape[0]
        T = dynamic.shape[1]
        
        h_pop = torch.mm(pop, self.W_pop)
        h_pop = torch.cat([h_pop.unsqueeze(1).repeat(1, N, 1), h_pop.unsqueeze(0).repeat(N, 1, 1)], dim=2)
        d_pop = torch.sigmoid(h_pop @ self.a_pop).reshape(N, N)
        h_demo = torch.mm(demo, self.W_demo)
        h_demo = torch.cat([h_demo.unsqueeze(1).repeat(1, N, 1), h_demo.unsqueeze(0).repeat(N, 1, 1)], dim=2)
        d_demo = torch.sigmoid(h_demo @ self.a_demo).reshape(N, N)
        h_eco = torch.mm(eco, self.W_eco)
        h_eco = torch.cat([h_eco.unsqueeze(1).repeat(1, N, 1), h_eco.unsqueeze(0).repeat(N, 1, 1)], dim=2)
        d_eco = torch.sigmoid(h_eco @ self.a_eco).reshape(N, N)
        h_geo = geo @ self.W_geo
        d_geo = torch.sigmoid(h_geo @ self.a_geo).reshape(N, N)
        dist = d_pop + d_demo + d_eco + d_geo
        dist = torch.softmax(dist, dim=-1)
        
        
        cov_weights = self.covid_weight(dynamic[:, :, 0].unsqueeze(-1).reshape(N*T, 1)).reshape(N, T, 1)
        #claim_weights = self.claim_weight(dynamic[:, :, 1:1+20].reshape(N*T, 20)).reshape(N, T, 20)
        hos_weights = self.hos_weight(dynamic[:, :, 1:1+4].reshape(N*T, 4)).reshape(N, T, 4)
        #vac_weights = self.vac_weight(dynamic[:, :, 1+20+4:1+20+4+17].reshape(N*T, 17)).reshape(N, T, 17)
        total_weights = torch.cat([cov_weights, hos_weights], dim=-1)
        if h0 is None:
            h0 = torch.randn(1, N, self.rnn_dim).to(self.device)
        h, hn = self.rnn(total_weights*dynamic)
        
        h_att = h.reshape(N,1,T,self.rnn_dim).repeat(1,N,1,1)
        h_att = h + (h_att * dist.reshape(N,N,1,1)).sum(1)
        y = self.linear(h_att)
        y = self.linear_2(F.leaky_relu(y))
        return y, [dist, total_weights, hn]
    
class HOIST_with_claim(nn.Module):
    def __init__(self, dynamic_dims, static_dims, rnn_dim, device):
        super(HOIST_with_claim, self).__init__()
        self.dynamic_dims = dynamic_dims
        self.static_dims = static_dims
        self.device = device
        self.rnn_dim = rnn_dim
        
        self.covid_weight = nn.Sequential(nn.Linear(1, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 1), nn.Sigmoid())
        self.claim_weight = nn.Sequential(nn.Linear(20, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 20), nn.Sigmoid())
        self.hos_weight = nn.Sequential(nn.Linear(4, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 4), nn.Sigmoid())
        self.vac_weight = nn.Sequential(nn.Linear(17, rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, 17), nn.Sigmoid())
        
        self.rnn = nn.LSTM(dynamic_dims, rnn_dim, batch_first=True)
        
        self.linear = nn.Linear(rnn_dim, rnn_dim)
        self.linear_2 = nn.Linear(rnn_dim, 1)
        
        pop_dim, demo_dim, eco_dim = static_dims
        self.W_pop = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(pop_dim, pop_dim).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_pop = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*pop_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.W_demo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(demo_dim, demo_dim).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_demo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*demo_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.W_eco = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(eco_dim, eco_dim).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_eco = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*eco_dim, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.W_geo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2, 2).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
        self.a_geo = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
    
    def forward(self, dynamic, static, h0=None):
        pop, demo, eco, geo = static
        N = pop.shape[0]
        T = dynamic.shape[1]
        
        h_pop = torch.mm(pop, self.W_pop)
        h_pop = torch.cat([h_pop.unsqueeze(1).repeat(1, N, 1), h_pop.unsqueeze(0).repeat(N, 1, 1)], dim=2)
        d_pop = torch.sigmoid(h_pop @ self.a_pop).reshape(N, N)
        h_demo = torch.mm(demo, self.W_demo)
        h_demo = torch.cat([h_demo.unsqueeze(1).repeat(1, N, 1), h_demo.unsqueeze(0).repeat(N, 1, 1)], dim=2)
        d_demo = torch.sigmoid(h_demo @ self.a_demo).reshape(N, N)
        h_eco = torch.mm(eco, self.W_eco)
        h_eco = torch.cat([h_eco.unsqueeze(1).repeat(1, N, 1), h_eco.unsqueeze(0).repeat(N, 1, 1)], dim=2)
        d_eco = torch.sigmoid(h_eco @ self.a_eco).reshape(N, N)
        h_geo = geo @ self.W_geo
        d_geo = torch.sigmoid(h_geo @ self.a_geo).reshape(N, N)
        dist = d_pop + d_demo + d_eco + d_geo
        dist = torch.softmax(dist, dim=-1)
        
        
        cov_weights = self.covid_weight(dynamic[:, :, 0].unsqueeze(-1).reshape(N*T, 1)).reshape(N, T, 1)
        claim_weights = self.claim_weight(dynamic[:, :, 1:1+20].reshape(N*T, 20)).reshape(N, T, 20)
        hos_weights = self.hos_weight(dynamic[:, :, 1+20:1+20+4].reshape(N*T, 4)).reshape(N, T, 4)
        vac_weights = self.vac_weight(dynamic[:, :, 1+20+4:1+20+4+17].reshape(N*T, 17)).reshape(N, T, 17)
        total_weights = torch.cat([cov_weights, claim_weights, hos_weights, -vac_weights], dim=-1)
        if h0 is None:
            h0 = torch.randn(1, N, self.rnn_dim).to(self.device)
        h, hn = self.rnn(total_weights*dynamic)
        
        h_att = h.reshape(N,1,T,self.rnn_dim).repeat(1,N,1,1)
        h_att = h + (h_att * dist.reshape(N,N,1,1)).sum(1)
        y = self.linear(h_att)
        y = self.linear_2(F.leaky_relu(y))
        return y, [dist, total_weights, hn]
    
class HOIST(nn.Module):
    def __init__(self, dynamic_dims, static_dims = None, distance_dims = None, rnn_dim=128, signs=None, device='cpu'):
        """The HOIST Model
        Args:
            dynamic_dims: List of integers (Number of features in each dynamic feature category, e.g., vaccination, hospitalization, etc.).
            static_dims (Optional): List of integers (Number of features in each static feature category, e.g., demographic, economic, etc.). If None, no static features are used.
            distance_dims (Optional): Interger (Number of distance types, e.g., geographical, mobility, etc.). If None, no distance features are used.
            rnn_dim: Integer (Number of hidden units in the RNN layer).
            signs: List of 1 or -1 (Field direction of each dynamic feature category, e.g., -1 for vaccination, +1 for hospitalization, etc.). If None, all signs are positive.
            device: String (Device to run the model, e.g., 'cpu' or 'cuda').
            
        Inputs:
            dynamic: List of FloatTensor with shape (N, T, D_k) (Dynamic features). D_k is the number of features in the k-th category and it should be the same as the k-th dimension in dynamic_dims.
            static (Optional): List of FloatTensor with shape (N, D_k) (Static features). D_k is the number of features in the k-th category and it should be the same as the k-th dimension in static_dims.
            distance (Optional): FloatTensor with shape (N, N, D_k) (Distance features). D_k is the number of distance types and it should be the same as the dimension in distance_dims.
            *** if both static and distance is None, the spatial relationships won't be used. ***
            h0 (Optional): FloatTensor with shape (1, N, rnn_dim) (Initial hidden state of the RNN layer). If None, it will be initialized as a random tensor.
        """
        
        super(HOIST, self).__init__()
        self.dynamic_dims = dynamic_dims
        self.dynamic_feats = len(dynamic_dims)
        self.static_dims = static_dims
        self.distance_dims = distance_dims
        self.device = device
        self.rnn_dim = rnn_dim
        self.signs = signs
        if self.signs != None:
            try:
                assert len(self.signs) == self.dynamic_feats
                assert all([s == 1 or s == -1 for s in self.signs])
            except:
                raise ValueError('The signs should be a list of 1 or -1 with the same length as dynamic_dims.')
        
        self.dynamic_weights = nn.ModuleList([nn.Sequential(nn.Linear(self.dynamic_dims[i], rnn_dim), nn.LeakyReLU(), nn.Linear(rnn_dim, self.dynamic_dims[i]), nn.Sigmoid()) for i in range(self.dynamic_feats)])
        
        self.total_feats = np.sum(self.dynamic_dims)       
        self.rnn = nn.LSTM(self.total_feats, rnn_dim, batch_first=True)
        
        self.linear = nn.Linear(rnn_dim, rnn_dim)
        self.linear_2 = nn.Linear(rnn_dim, 1)
        
        self.static_dims = static_dims
        if self.static_dims != None:
            self.static_feats = len(static_dims)
    
            self.w_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.static_dims[i], self.static_dims[i]).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device) for i in range(self.static_feats)])
            self.a_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.Tensor(2*self.static_dims[i], 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device) for i in range(self.static_feats)])

        if self.distance_dims != None:
            self.W_dis = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(distance_dims, distance_dims).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
            self.a_dis = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(distance_dims, 1).type(torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True).to(device)
    
    def forward(self, dynamic, static = None, distance = None, h0 = None):
        try:
            assert len(dynamic) == self.dynamic_feats
        except:
            print('The number of dynamic features is not correct.')
            return None
        if self.static_dims != None:
            try:
                assert len(static) == self.static_feats
            except:
                print('The number of static features is not correct.')
                return None
        if self.distance_dims != None:
            try:
                assert distance.shape[2] == self.distance_dims
            except:
                print('The number of distance features is not correct.')
                return None
        
        static_dis = []
        N = dynamic[0].shape[0]
        T = dynamic[0].shape[1]
        if self.static_dims != None:
            for i in range(self.static_feats):
                h_i = torch.mm(static[i], self.w_list[i])
                h_i = torch.cat([h_i.unsqueeze(1).repeat(1, N, 1), h_i.unsqueeze(0).repeat(N, 1, 1)], dim=2)
                d_i = torch.sigmoid(h_i @ self.a_list[i]).reshape(N, N)
                static_dis.append(d_i)

        if self.distance_dims != None:
            h_i = distance @ self.W_dis
            h_i = torch.sigmoid(h_i @ self.a_dis).reshape(N, N)
            static_dis.append(h_i)
            
        if self.static_dims != None or self.distance_dims != None:
            static_dis = torch.stack(static_dis, dim=0)
            static_dis = static_dis.sum(0)
            static_dis = torch.softmax(static_dis, dim=-1)
        
        dynamic_weights = []
        for i in range(self.dynamic_feats):
            cur_weight = self.dynamic_weights[i](dynamic[i].reshape(N*T, -1)).reshape(N, T, -1)
            if self.signs != None:
                cur_weight = cur_weight * self.signs[i]
            dynamic_weights.append(cur_weight)
        dynamic_weights = torch.cat(dynamic_weights, dim=-1)

        if h0 is None:
            h0 = torch.randn(1, N, self.rnn_dim).to(self.device)
        dynamic = torch.cat(dynamic, dim=-1)
        h, hn = self.rnn(dynamic_weights*dynamic)
        
        if self.static_dims != None or self.distance_dims != None:
            h_att = h.reshape(N,1,T,self.rnn_dim).repeat(1,N,1,1)
            h = h + (h_att * static_dis.reshape(N,N,1,1)).sum(1)
        y = self.linear(h)
        y = self.linear_2(F.leaky_relu(y))
        return y, [static_dis, dynamic_weights, hn]
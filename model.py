
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

    
class HOIST(nn.Module):
    def __init__(self, dynamic_dims, static_dims, rnn_dim, device):
        super(HOIST, self).__init__()
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from collections import defaultdict
import math
import pickle
import random
import argparse

class InequalityEmbedding(nn.Module):

    def __init__(self, poi_num, cbg_num, percentile_num, embedding_dim, device='cpu'):
        # poi_num: int, number of POIs in the MSA
        # cbg_num: int, numebr of CBGs in the MSA
        # percentile_num: int, number of treatment levels to learn embedding
        # embedding_dim: int
        super(InequalityEmbedding, self).__init__()
        self.device = device
        self.category_num = 4 # 4 categories
        self.poi_num = poi_num
        self.cbg_num = cbg_num
        self.feature_num = 5 # 5 demographic features
        self.percentile_num = percentile_num
        self.embedding_dim = embedding_dim
        self.cate_embeddings = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim).to(device)
        self.poi_embeddings = nn.Embedding(num_embeddings=poi_num, embedding_dim=embedding_dim).to(device)
        self.cbg_embeddings = nn.Embedding(num_embeddings=cbg_num, embedding_dim=embedding_dim).to(device)
        self.percentile_embeddings = nn.Embedding(num_embeddings=5 * self.percentile_num, embedding_dim=embedding_dim).to(device)
    
    def pick_feat(self, feature_id, feature_val):
        # center of each treatment level, [10%, 30%, 50%, 70%, 90%] for percentile_num = 5
        percentiles = torch.tensor([1/self.percentile_num/2+i/self.percentile_num for i in range(self.percentile_num)], device=self.device)
        feature_val = torch.unsqueeze(feature_val,1)
        # weights assigned to each treatment level
        m = F.softmax(-torch.abs(self.percentile_num * (feature_val - percentiles)), dim=1)
        # embedding of a specific feature_val
        feat_emb = torch.zeros((feature_val.shape[0], self.embedding_dim), device=self.device)
        for i in range(self.percentile_num):
            curr_emb = self.percentile_embeddings(torch.tensor(feature_id * self.percentile_num + i + 0*feature_val.squeeze()).long())
            feat_emb += m[:, i:i+1] * curr_emb
        return feat_emb

    def init(self):
        # initialize embedding vectors
        xavier_normal_(self.cate_embeddings.weight.data)
        xavier_normal_(self.poi_embeddings.weight.data)
        xavier_normal_(self.cbg_embeddings.weight.data)
        xavier_normal_(self.percentile_embeddings.weight.data)

    def regularization_feat(self):
        # demographic regularization term: sum of square distance between adjacent treatment level embeddings
        l = torch.tensor(0, dtype=torch.float32).to(self.device)
        for f in range(self.feature_num):
            for i in range(self.percentile_num-1):
                ff = torch.tensor(f, dtype=torch.long).to(self.device)
                ii = torch.tensor(i, dtype=torch.long).to(self.device)
                a0 = self.percentile_embeddings(ff*self.percentile_num+ii)
                a1 = self.percentile_embeddings(ff*self.percentile_num+ii+1)
                l += torch.sum(torch.square(a1-a0))
        return l

    def regularization_CBG(self, CBG_neighbors):
        # spatial regularization term: weighted sum of square distance between adjacent CBG's embeddings
        c1 = CBG_neighbors[:, 0].long()
        c2 = CBG_neighbors[:, 1].long()
        d = CBG_neighbors[:, 2]
        a0 = self.cbg_embeddings(c1)
        a1 = self.cbg_embeddings(c2)
        return torch.sum(torch.sum(torch.square(a1-a0), 1)*d)
        
    def forward(self, inputs):
        # embedding loss
        cate_id = inputs[:, 0].long()
        poi_id = inputs[:, 1].long()
        cbg_id = inputs[:, 2].long()
        feature0 = inputs[:, 3]
        feature1 = inputs[:, 4]
        feature2 = inputs[:, 5]
        feature3 = inputs[:, 6]
        feature4 = inputs[:, 7]
        altcbg0 = inputs[:, 8].long()
        altfeature0 = inputs[:, 9]
        altcbg1 = inputs[:, 10].long()
        altfeature1 = inputs[:, 11]
        altcbg2 = inputs[:, 12].long()
        altfeature2 = inputs[:, 13]
        altcbg3 = inputs[:, 14].long()
        altfeature3 = inputs[:, 15]
        altcbg4 = inputs[:, 16].long()
        altfeature4 = inputs[:, 17]
        cate_embed = self.cate_embeddings(cate_id)
        poi_embed = self.poi_embeddings(poi_id)
        cbg_embed = self.cbg_embeddings(cbg_id)
        observed_embed0 = self.pick_feat(0, feature0)
        observed_embed1 = self.pick_feat(1, feature1)
        observed_embed2 = self.pick_feat(2, feature2)
        observed_embed3 = self.pick_feat(3, feature3)
        observed_embed4 = self.pick_feat(4, feature4)
        altcbg_embed0 = self.cbg_embeddings(altcbg0)
        altcbg_embed1 = self.cbg_embeddings(altcbg1)
        altcbg_embed2 = self.cbg_embeddings(altcbg2)
        altcbg_embed3 = self.cbg_embeddings(altcbg3)
        altcbg_embed4 = self.cbg_embeddings(altcbg4)
        altfeat_embed0 = self.pick_feat(0, altfeature0)
        altfeat_embed1 = self.pick_feat(1, altfeature1)
        altfeat_embed2 = self.pick_feat(2, altfeature2)
        altfeat_embed3 = self.pick_feat(3, altfeature3)
        altfeat_embed4 = self.pick_feat(4, altfeature4)
        
        loss_cate_cbg = -F.logsigmoid(torch.sum(cate_embed * cbg_embed, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(cate_embed * altcbg_embed0, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(cate_embed * altcbg_embed1, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(cate_embed * altcbg_embed2, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(cate_embed * altcbg_embed3, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(cate_embed * altcbg_embed4, dim=1))
        loss_cate_feat = -F.logsigmoid(torch.sum(cate_embed * observed_embed0, dim=1)) \
            -F.logsigmoid(torch.sum(cate_embed * observed_embed1, dim=1)) \
            -F.logsigmoid(torch.sum(cate_embed * observed_embed2, dim=1)) \
            -F.logsigmoid(torch.sum(cate_embed * observed_embed3, dim=1)) \
            -F.logsigmoid(torch.sum(cate_embed * observed_embed4, dim=1)) \
            - F.logsigmoid(-torch.sum(cate_embed * altfeat_embed0, dim=1)) \
            - F.logsigmoid(-torch.sum(cate_embed * altfeat_embed1, dim=1)) \
            - F.logsigmoid(-torch.sum(cate_embed * altfeat_embed2, dim=1)) \
            - F.logsigmoid(-torch.sum(cate_embed * altfeat_embed3, dim=1)) \
            - F.logsigmoid(-torch.sum(cate_embed * altfeat_embed4, dim=1))
        loss_poi_cbg = -F.logsigmoid(torch.sum(poi_embed * cbg_embed, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(poi_embed * altcbg_embed0, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(poi_embed * altcbg_embed1, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(poi_embed * altcbg_embed2, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(poi_embed * altcbg_embed3, dim=1)) \
            - 0.2 * F.logsigmoid(-torch.sum(poi_embed * altcbg_embed4, dim=1))
        loss_poi_feat = -F.logsigmoid(torch.sum(poi_embed * observed_embed0, dim=1)) \
            -F.logsigmoid(torch.sum(poi_embed * observed_embed1, dim=1)) \
            -F.logsigmoid(torch.sum(poi_embed * observed_embed2, dim=1)) \
            -F.logsigmoid(torch.sum(poi_embed * observed_embed3, dim=1)) \
            -F.logsigmoid(torch.sum(poi_embed * observed_embed4, dim=1)) \
            - F.logsigmoid(-torch.sum(poi_embed * altfeat_embed0, dim=1)) \
            - F.logsigmoid(-torch.sum(poi_embed * altfeat_embed1, dim=1)) \
            - F.logsigmoid(-torch.sum(poi_embed * altfeat_embed2, dim=1)) \
            - F.logsigmoid(-torch.sum(poi_embed * altfeat_embed3, dim=1)) \
            - F.logsigmoid(-torch.sum(poi_embed * altfeat_embed4, dim=1))
        return torch.sum(loss_cate_cbg)+torch.sum(loss_cate_feat)+torch.sum(loss_poi_cbg)+torch.sum(loss_poi_feat)

# set random seeds
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# prepare data for spatial regularization
def prepare_spatial_regularization(city_id, threshold, device):
    dists = np.load('data/CBG_dists_'+str(city_id)+'.npy')
    a, b = np.where(dists < threshold)
    CBG_neighbors = np.array([np.array([a[i], b[i], dists[a[i]][b[i]]]) for i in range(len(a)) if a[i] < b[i]])
    def kernel(x, threshold):
        return np.exp(-np.square(x)/2/np.square(threshold))/2/np.pi/np.square(threshold)
    CBG_neighbors[:, 2] = kernel(CBG_neighbors[:, 2], threshold)
    CBG_neighbors = torch.tensor(CBG_neighbors).to(device)
    return CBG_neighbors

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-embedding_dim', default = 16, type=int) # dimension of embedding vectors
    parser.add_argument('-batch_size', default=512, type=int) # batch size
    parser.add_argument('-city_id', default = 1, type=int) # MSA ID, 1: New York, 2: Los Angeles, 3: Chicago
    parser.add_argument('-cuda', default = 6, type=int) # GPU device
    parser.add_argument('-seed', default = 1999, type=int) # random seed
    parser.add_argument('-percentile_num', default = 5, type=int) # number of treatment levels to learning embedding
    parser.add_argument('-threshold', default = 2.5, type=float) # spatial threshold
    parser.add_argument('-l2', default = 0.01, type=float) # strength of demographic regulariztion
    parser.add_argument('-l2_CBG', default = 0.01, type=float) # strength of spatial regularization
    args = parser.parse_args()
    
    setup_seed(args.seed)  
    device = torch.device('cuda:'+str(args.cuda))
    
    # Mapping CBG ID & POI ID to embedding IDs
    with open('data/cbgid_embeddingid_mapping.pkl', 'rb') as f:
        cbgid_dict = pickle.load(f)
    with open('data/poiid_embeddingid_mapping.pkl', 'rb') as f:
        poiid_dict = pickle.load(f)
    poiid_mapping = poiid_dict[args.city_id]
    poi_num = len(poiid_mapping)
    cbgid_mapping = cbgid_dict[args.city_id]
    cbg_num = len(cbgid_mapping)
    del poiid_dict
    del cbgid_dict

    dataset = torch.load('data/dataset'+str(args.city_id)+'.pt', map_location=device)
    lendata = dataset.size()[0]
    CBG_neighbors = prepare_spatial_regularization(args.city_id, args.threshold, device)
    
    losses = []
    model = InequalityEmbedding(poi_num, cbg_num, args.percentile_num, args.embedding_dim, device)
    model.to(device)
    model.init()
    l2 = torch.tensor(args.l2).to(device)
    l2_CBG = torch.tensor(args.l2_CBG).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    epochs = 10001
    print("epoch\t loss\t")
    weights = {}
    batch_size = args.batch_size

    # train embeddings
    for epoch in range(epochs):
        total_loss = 0
        sim_loss = 0
        feat_loss = 0
        CBG_loss = 0
        for i in range(math.ceil(lendata/batch_size)):
            data = dataset[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            # embedding loss
            loss_similarity = model(data)
            # demographic regularization
            loss_feat_regularization = model.regularization_feat() * data.size()[0]
            # spatial regularization
            loss_CBG_regularization = model.regularization_CBG(CBG_neighbors) * data.size()[0]
            loss = loss_similarity + l2 * loss_feat_regularization + l2_CBG * loss_CBG_regularization
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / dataset.size()[0]
            sim_loss += loss_similarity.item() / dataset.size()[0]
            feat_loss += loss_feat_regularization.item() / dataset.size()[0]
            CBG_loss += loss_CBG_regularization.item() / dataset.size()[0]
        if epoch % 50 == 0:
            print(epoch, total_loss, sim_loss, feat_loss, CBG_loss)
        if epoch % 1000 == 0:
            # save embedding vectors every 1000 epoch
            weights[epoch] = [model.cate_embeddings.weight.to('cpu').clone().detach(), model.poi_embeddings.weight.to('cpu').clone().detach(), model.cbg_embeddings.weight.to('cpu').clone().detach(), model.percentile_embeddings.weight.to('cpu').clone().detach()]
        losses.append(total_loss)

    write_dict = {}
    write_dict['weights'] = weights
    write_dict['losses'] = losses
    write_dict['args'] = args

    outfilename = 'data/embedding_city'+str(args.city_id)+'dim'+str(args.embedding_dim)+'batch'+str(args.batch_size)+'seed'+str(args.seed)+'percentile'+str(args.percentile_num)+'threshold'+str(args.threshold)+'l2'+str(args.l2)+'l2_CBG'+str(args.l2_CBG)+'.pkl'

    with open(outfilename, 'wb') as f:
        pickle.dump(write_dict, f)


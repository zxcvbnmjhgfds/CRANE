import numpy as np
import pandas as pd
import pickle
import random
import torch
from torch.nn.functional import softmax
from collections import defaultdict
import argparse

# transform random walks to torch datasets
def transform_dataset(sample_dict, city_id):
    with open('./data/cbg_locate_in_msa_dict.pkl', 'rb') as f:
        cbg_locate_in_msa_dict = pickle.load(f)
    cbgs = [x for x in cbg_locate_in_msa_dict.keys() if cbg_locate_in_msa_dict[x] == city_id]
    # CBG features
    df = pd.read_csv('./data/middle.csv')
    df = df[df.census_block_group.isin(cbgs)]
    df = df[df['Total population'] > 100]
    df = df[['census_block_group', 'female_ratio', 'white_ratio', 'bachelor_ratio', 'average_income', 'population_young_ratio', 'population_old_ratio', 'disability_household_ratio']]
    df = df.dropna()
    features = ['female_ratio', 'white_ratio', 'bachelor_ratio','average_income', 'disability_household_ratio']
    
    def interpret(x):
        # category_id, POI_id, observed CBG_id
        g = [x[0]-1, x[1]-1, x[2]-1]
        # percentile of 5 observed outcome 
        for k in range(5):
            temp = df[features[k]]
            g.append(np.sum(temp <= x[3+k]) / len(df))
        for k in range(5):
            # alternative CBG_id
            g.append(x[8+2*k]-1)
            temp = df[features[k]]
            # percentile of alternative outcome 
            g.append(np.sum(temp <= x[9+2*k]) / len(df))
        return g
    dataset = []
    for cate in range(1, 5):
        dataset += [interpret(x) for x in sample_dict[cate]]
    random.shuffle(dataset)
    dataset = torch.tensor(dataset)
    torch.save(dataset, './data/dataset'+str(city_id)+'.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-poi_num', default = 1, type=int) # Number of POIs sampled from the category in each iteration
    parser.add_argument('-city_id', default = 1, type=int) # MSA ID, 1: New York, 2: Los Angeles, 3: Chicago
    parser.add_argument('-cuda', default = 4, type=int) # CUDA device ID
    parser.add_argument('-q', default=5, type=int) # Number of covariate's strata
    args = parser.parse_args()
    
    # Load ID of CBGs located in the MSA
    with open('./data/cbg_locate_in_msa_dict.pkl', 'rb') as f:
        cbg_locate_in_msa_dict = pickle.load(f)
    city_cbgs = [x for x in cbg_locate_in_msa_dict.keys() if cbg_locate_in_msa_dict[x] == args.city_id]
    del cbg_locate_in_msa_dict

    # Load demographic features of CBGs located in the MSA
    df = pd.read_csv('./data/middle.csv')
    df = df[df.census_block_group.isin(city_cbgs)]
    df = df[['census_block_group', 'female_ratio', 'white_ratio', 'bachelor_ratio', 'average_income', 'population_young_ratio', 'population_old_ratio', 'disability_household_ratio']]
    df = df.dropna()
    CBG_record = list(df.census_block_group)

    # Visit patterns (CBG-POI network) in the MSA
    # visit_each_poi_new: dict
    # keys: POI ID (str, 'sg:4433120bada9409da78492d31f6bbc9e')
    # values: (category ID (int, 1~4), CBG visit dictionary(dict))
    # CBG visit dictionary: 
    # keys: CBG ID (int, 481576747003)
    # values: CBG's urban resource accessibility to the POI 
    with open('./data/visit_each_poi_new_'+str(args.city_id)+'.pkl', 'rb') as f:
        visit_each_poi_new = pickle.load(f)
    
    # Mapping CBG ID & POI ID to embedding IDs
    with open('./data/cbgid_embeddingid_mapping.pkl', 'rb') as f:
        cbgid_dict = pickle.load(f)
    with open('./data/poiid_embeddingid_mapping.pkl', 'rb') as f:
        poiid_dict = pickle.load(f)
    cbgid_dict = cbgid_dict[args.city_id]
    poiid_dict = poiid_dict[args.city_id]

    poi_by_cate = dict()
    for cate in range(1, 5):
        # all POIs within the category
        poi_in_cate = [x for x in visit_each_poi_new.keys() if visit_each_poi_new[x][0] == cate and poiid_dict[x] > 0]
        # sum of CBG's urban resource accessibility to each POI
        poi_num_dict = [sum(visit_each_poi_new[x][1].values()) for x in poi_in_cate]
        poi_by_cate[cate] = (poi_in_cate, poi_num_dict)

    # sample POIs from a category Q based on POI's total urban resource accessibility
    def sample_poi(cate, num=1, cu='cuda:7'):
        device = torch.device(cu)
        a, prob = poi_by_cate[cate]
        prob = torch.tensor(prob, device=device)
        prob = prob / torch.sum(prob)
        idx = prob.multinomial(num_samples=num,  replacement = True).to('cpu')
        return np.array(a)[idx.numpy()]
    
    # sample a CBG from all CBGs that have visited a POI based on their urban resource accessibility to the POI
    def sample_cbg_normalized(poi_id, num=1, cu='cuda:7'):
        device = torch.device(cu)
        visit_dict = visit_each_poi_new[poi_id][1]
        a = [x for x in list(visit_dict.keys()) if cbgid_dict[x] > 0]
        if len(a) == 0:
            return None
        prob = np.array([visit_dict[x] for x in a])
        prob = torch.tensor(prob, device=device)
        prob = prob / torch.sum(prob)
        idx = prob.multinomial(num_samples=num, replacement = True).to('cpu')
        return np.array(a)[idx.numpy()]

    features = ['female_ratio', 'white_ratio', 'bachelor_ratio', 'average_income',
    'disability_household_ratio', 'population_young_ratio', 'population_old_ratio']
    # stratified covariates to be matched when sampling alternative outcome
    def confounder(f):
        return ['binned_'+x for x in features if x != f]
    # stratify demographic features
    for f in features:
        binned_name = 'binned_'+f
        _, bins = np.array(pd.qcut(df[f], q=args.q, duplicates='drop', retbins = True))
        df[binned_name] = np.array(pd.qcut(df[f], q=args.q, labels = list(range(len(bins)-1)),duplicates='drop', retbins = False))
    
    # sample observed and alternative outcome
    def sample_feature(cbg_id, poi_id, feat, cu='cuda:7'):
        visit_dict = visit_each_poi_new[poi_id][1]
        # all CBGs that have visited the POI
        visited_cbgs = [x for x in list(visit_dict.keys()) if cbgid_dict[x] > 0]
        # observed outcome and covariates of the observed CBG
        observed_outcome = df[df.census_block_group == cbg_id][feat].values[0]
        conf = df[df.census_block_group == cbg_id][confounder(feat)].values[0]
        # sample an alternative CBG from CBGs that have not visited the POI
        middle = df[~df.census_block_group.isin(visited_cbgs)]
        vals = np.array(list(middle[feat]))
        cbgs = list(middle['census_block_group'])
        # select the alternative CBG based on discrepancy in stratified covariates
        dists = np.sum(np.square(np.array(middle[confounder(feat)]) - conf), axis=1)
        prob = torch.tensor(dists, dtype=torch.float64, device=cu)
        idx = softmax(-10*prob, dim=0).multinomial(num_samples=1,replacement = True).to('cpu')
        alternative_outcomes = vals[idx.numpy()]
        alternative_cbg = [cbgs[i] for i in idx.numpy()]
        # observed CBG, observed outcome, alternative CBG, alternative outcome
        return cbg_id, observed_outcome, alternative_cbg[0], alternative_outcomes[0]

    # category IDs: 
    # 1: Art & Recreation, 2: Sports, 3: Education, 4: Health
    sample_dict = {1: [], 2: [], 3:[], 4:[]}
    for cate in range(1, 5):
        sampled_POIs = sample_poi(cate, num=args.poi_num, cu='cuda:'+str(args.cuda))
        for poi in sampled_POIs:
            CBGs = sample_cbg_normalized(poi, num=1, cu='cuda:'+str(args.cuda))
            for cbg_id in CBGs:
                if cbg_id not in CBG_record:
                    continue
                sample = [cate, poiid_dict[poi], cbgid_dict[cbg_id]]
                alts = []
                # sample an alternative CBG for each of the five demographic features
                for feat in features[:5]:
                    _, observed_outcome, alt_cbg, alt_outcome = sample_feature(cbg_id, poi, feat, cu='cuda:'+str(args.cuda))
                    sample.append(observed_outcome)
                    alts.append(cbgid_dict[alt_cbg])
                    alts.append(alt_outcome)
                sample_dict[cate].append(sample+alts)

    outfilename = './data/CounterfactualRandomWalk_'+str(args.city_id)+'_poinum'+str(args.poi_num)+'.pkl'
    with open(outfilename, 'wb') as f:
        pickle.dump(sample_dict, f)
    
    transform_dataset(sample_dict, args.city_id)
# CRANE
Source code for *Counterfactual RANdom-walk based Embedding* (CRANE) algorithm in **Identifying Mobility Inequality in Urban Space with Large-scale Open Data**.

## Counterfactual Random Walk
### Code and File
*CounterfactualRandomWalk.py* is the code for counterfactual random walk on a POI-CBG visitation network. It requires *cbg_locate_in_msa_dict.pkl*, *middle.csv*, *visit_each_poi_new_3.pkl*, *cbgid_embeddingid_mapping.pkl*, *poiid_embeddingid_mapping.pkl* in the data folder to generate sampled random walks.
### Environment
The code is run with python 3.8.3 and torch 1.7.0.
### How to run
Run following code to generate random walks on Chicago's visitation network for following embedding algorithm. (*data/dataset3.pt*)
```
python CounterfactualRandomWalk.py -poi_num 200000 -city_id 3 -q 5
```

## Network Embedding for Mobility Inequality
### Code and File
*CRANE_embedding.py* is the code for network embedding algorithm on random walks. It takes *dataset3.pt* and *CBG_dists_3.npy* as inputs to generate embedding vectors in *embedding_city3dim64batch10000seed1999percentile5threshold2.5l20.01l2_CBG0.0001.pkl* in the data folder. 
### Environment
The code is run with python 3.8.3 and torch 1.7.0.
### How to run
Run following code to generate embedding vectors for POI categories, POIs, CBGs, and treatment levels in Chicago MSA.
```
python CRANE_embedding.py -embedding_dim 64 -batch_size 10000 -city_id 3 -seed 1999 -percentile_num 5 -threshold 2.5 -l2 0.01 -l2_CBG 0.0001
```

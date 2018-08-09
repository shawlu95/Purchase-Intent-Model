import pandas as pd
import scipy

def N_Nearest_Neighbor_for_Member(n, member_srl, df, target_cat = None, weight = None, verbose = False):
    if type(weight) == pd.core.series.Series:
        df = df * weight
        
    if type(target_cat) == list:
        df = df[target_cat]
    
    target = df.loc[[member_srl]]
    member_srls = df.index.values
    ranking = scipy.spatial.distance.cdist(target.iloc[:,:], df.iloc[:,:], metric='euclidean')
    scores = pd.DataFrame({"member_srl" : member_srls, "dist": ranking[0]})
    scores = scores.set_index('member_srl')
    sorted_ = scores.sort_values(by=['dist'])
    
    if verbose:
        print(sorted_.iloc[:n])
    
    return sorted_.index.values[:n]

def N_Nearest_Neighbors_for_Member(n, member_srl, df, target_cat = None, weight = None, verbose = False, metric = "euclidean"):
    if type(weight) == pd.core.series.Series:
        df = df * weight
        
    if type(target_cat) == list:
        df = df[target_cat]
    
    target = df.loc[[member_srl]]
    member_srls = df.index.values
    ranking = scipy.spatial.distance.cdist(target.iloc[:,:], df.iloc[:,:], metric=metric)
    scores = pd.DataFrame({"member_srl" : member_srls, "dist": ranking[0]})
    scores = scores.set_index('member_srl')
    sorted_ = scores.sort_values(by=['dist'])
    
    if verbose:
        print(sorted_.iloc[:n])
    
    return sorted_[:n]
# -*- coding: utf-8 -*-

from pyclustering.cluster.clarans import clarans
import pandas as pd
import numpy as np
import pickle
from clust_fct import moy_evaluation

def Clarans(uim,nb_clusters,nb_iter,maxneighbor):
    
    data = uim.values.tolist()
    clarans_instance = clarans(data,
                               nb_clusters,nb_iter ,
                               maxneighbor)
    print("here")
    #calls the clarans method 'process' to implement the algortihm
    # (ticks, result) = timedcall(clarans_instance.process())
    # print("Execution time : ", ticks, "\n")
    clarans_instance.process()
    #returns the clusters 
    c = clarans_instance.get_clusters()
    
    #returns the mediods 
    medoids = clarans_instance.get_medoids()
    
    # for i in (0,1,2):
    #     for j in range(len(c[i])):
    #         c[i][j] += 1
       
    #les clusters de 
    
    
    names = uim.index
    
    index = c[0] + c[1] + c[2]
    
    cliste = []
    
    for i in index:
        cliste.append(names[i])
        
    data = [0]*len(c[0]) + [1]*len(c[1]) + [2]*len(c[2])
    
    clusters = pd.DataFrame(index = cliste, data = {'cluster': data })

    return clusters,medoids


def start_clarans(data_path,pmf_path,formule = 'user-user',nb_clusters = 3,
                  nb_iter = 7,maxneighbor = 4):
    
    #lecture des donnees
    d = pd.read_csv(data_path)
    uim = d.pivot_table("rating","userId","movieId")
    
    uim.fillna(0,inplace = True)
    
    #matrice pmf
    pmf = pd.read_pickle(pmf_path)
    pmf.columns.name = 'movieId'
    pmf.index.name = 'userId'
    
    #preparation des donnees pour le clustering
    uim1  = uim.apply(func = lambda x:x.fillna(x.mean()),axis =1)
    #val = uim1.values
    
    if formule == 'item-item':
        df = pmf.T
        with open("item-item_clarans_clusters_pmf_374.pkl",'rb') as f:
            u = pickle.Unpickler(f)
            clusters = u.load()
        # pivot_pmf = pmf.T
        # val = pivot_pmf.values
        # val = val.tolist()
    elif formule == 'user-user':
        df = pmf
        with open('user-userclarans_clusters_pmf_374.pkl','rb') as f:
            u = pickle.Unpickler(f)
            clusters = u.load()
        # val = pmf.values
        # val = val.tolist()
    else:
        raise ValueError("valeur de formule est soit \
                         \'user\-user' ou \'item-item\'") 
                         

    print("debut clustering ...")
    # clusters,m = Clarans(df,nb_clusters,nb_iter,maxneighbor)
    
    


    # print("sauvgarde des clusters...")
    # with open(formule+"clarans_clusters_pmf_374.pkl",'wb') as f:
    #     p = pickle.Pickler(f)
    #     p.dump(clusters)   
        

    moy_p,moy_r,users_pred = moy_evaluation(uim,clusters,formule,
                                 matrice=pmf)
    
 
    
    #enregistrement
    name = "clarans_"+formule+"_moy_prec_.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_p)
        
    name = "clarans_"+formule+"_moy_rec_.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_r)
        
    name = "clarans_"+formule+"_users_pred.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(users_pred)


    
    return moy_p,moy_r


#start calarans with user-user

pmf_path = "pmf_MovieLens50.pkl"

moy_p,moy_r = start_clarans("MovieLens_new.csv",pmf_path,
                            formule = 'item-item')
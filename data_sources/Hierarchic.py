# -*- coding: utf-8 -*-

import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from sklearn import preprocessing
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle
from clust_fct import moy_evaluation

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def elbow(Z):
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print ("clusters:", k)

def hierarchic(data,max_d = 365,p = 5,affich_dendo = False):   
    X = data.values
    names = data.index
    
    # Centrage et Réduction:
    
    # std_scale = preprocessing.StandardScaler().fit(X)
    # X_scaled = std_scale.transform(X)
    
    # Clustering hiérarchique:
    
    # Z = linkage(X_scaled, 'ward')
    
    Z = linkage(X, 'ward')
    
    
    
    c, coph_dists = cophenet(Z, pdist(X))
    print(" C = ",c)
    
    
    #affichage du dendogramme:
    
    if affich_dendo == True : 
        fancy_dendrogram(Z,p= p,truncate_mode= 'lastp',
                       leaf_rotation=90.,
                       leaf_font_size=12.,
                       show_contracted=True,
                       max_d = max_d
                   )
    
    #plot_dendrogram(Z, names)

    clusters = fcluster(Z,max_d,criterion='distance')
        
    result = pd.DataFrame(data=clusters,index = names,columns=['cluster'])
    
    taille_cluster = result.cluster.value_counts()
    print("taille des clusters :\n",taille_cluster)
    
    return result,Z


def star_Hie(data_path,pmf_path,formule = 'user-user',max_d = 365,nb_clusters = None):
    #lecture des donnees
    d = pd.read_csv(data_path)
    uim = d.pivot_table("rating","userId","movieId")
    
    uim.fillna(0,inplace = True)
    
    #matrice pmf
    pmf = pd.read_pickle(pmf_path)
    pmf.columns.name = 'movieId'
    pmf.index.name = 'userId'
    
    #preparation des donnees pour le clustering
    if formule == 'item-item':
        pivot_pmf = pmf.T
        print("item-item")
        print(pivot_pmf.head())
        # max_d = 365
        clusters,Z = hierarchic(pivot_pmf,max_d,p = 10,affich_dendo = True)
    elif formule == 'user-user':
        clusters,Z = hierarchic(pmf,max_d,p = 10,affich_dendo = True)
    else:
        raise ValueError("valeur de formule est soit \
                         \'user\-user' ou \'item-item\'")
                         
    #uim1  = uim.apply(func = lambda x:x.fillna(x.mean()),axis =1)
    #val = uim1.values
    
    
    #debut evaluation
    moy_p,moy_r,users_pred = moy_evaluation(uim,clusters,formule,matrice = pmf)
    
    
    #enregistrement
    name = "hier_"+formule+"_moy_prec.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_p)
        
    name = "hier_"+formule+"_moy_rec.pkl"    
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_r)  
        
    name = "hier_"+formule+"_users_pred.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(users_pred)
    
    return moy_p,moy_r

#start hierarchic with user-user

# pmf_path = "pmf_MovieLens50.pkl"


# star_Hie("MovieLens_new.csv",pmf_path)


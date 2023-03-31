# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from collections import defaultdict
import pickle

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

def SVM(X_train,y_train, X_test, votes, kernel = 'linear'):
    #construction +entrainement
    svclassifier = SVC(kernel= kernel)
    svclassifier.fit(X_train, y_train)
    #prediction
    y_pred = svclassifier.predict(X_test)
    pred = pd.Series(index = votes.index,data = y_pred)

    return pred

def evaluation(votes,pred,k = 10):
 while(votes.shape[0] < k):
    k = int(input("on a moins de {} votes \n veuiilez entrer une nouvelle val de k".format(k)))
 #nb of all relevent items 
 relevent = sum(votes.values)
 non_relevent = len(votes) - relevent
 pred.sort_values(ascending = False,inplace = True)
 #on choisit les k meilleurs
 pred_top_k = pred.iloc[0:k-1].index
 # nb of relevent items in the selected nb_rec items
 relevent_selected = sum([elt in votes.loc[votes == 1].index for elt in pred_top_k])
 print('true relevent = ',relevent)
 print('relevent selected  = ',relevent_selected)
 pred.sort_values(inplace = True)
 pred_worst_k = pred.iloc[0:k-1].index
 non_relevent_selected = sum([elt in votes.loc[votes == 0].index for elt in pred_worst_k])
 print('true NON relevent = ',non_relevent)
 print('relevent selected  = ',non_relevent_selected)
 #precision 1: relevent ;;;; 0:non-relevent
 p1 = relevent_selected/k
 p0 = non_relevent_selected/k
 #recall
 r1 = relevent_selected/relevent if relevent != 0 else 0
 r0 = non_relevent_selected/non_relevent if non_relevent != 0 else 0
 print("-"*10)
 print("k = ",k) 
 print("precision : \n")
 print("  1 : ",p1)
 print("  0 : ",p0)
 print("recall : \n")
 print("  1 : ",r1)
 print("  0 : ",r0)
 print("-"*10)
 return p1,r1,p0,r0


def start_SVM(data_path, train_path, test_path, kernel = 'linear',
              liste_k =[10,15,20,30,40,50]):
    df = pd.read_csv(data_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    uim = df.pivot_table(index = 'userId',columns = 'movieId', values = 'rating')
    users = df.userId.unique()
    uim.fillna(0,inplace = True)
    
    # #### new  : pmf !!!!
    # uim = pd.read_pickle(data_path)
    # uim.columns.name = 'movieId'
    # uim.index.name = 'userId'
    # users = uim.index.values
    # ####end  new
    
    moy_precision_rel = defaultdict(float)
    moy_recall_rel = defaultdict(float)
    moy_precision_n_rel = defaultdict(float)
    moy_recall_n_rel = defaultdict(float)
    
    i = 1
    seul = 0
    moy_accuracy = 0
    taille = 0
    
    for u in users:
        print(u)
        # user_train = train.loc[train.userId == u]
        # movie_train = user_train.movieId.values
        # y_train = user_train.apply(func = lambda x: 1 if x['rating'] >= 3.5 else 0,
        #                            axis = 1)
        # y_train = y_train.values
        

        # X_train = []
        # for m in movie_train:
        #     X_train.append(uim.loc[:,m].values.tolist())
            
        # user_test = test.loc[test.userId == u]
        # movie_test = user_test.movieId.values
        # y_test = user_test.apply(func = lambda x: 1 if x['rating'] >= 3.5 else 0,
        #                            axis = 1)
        # y_test = y_test.values
        

        # X_test = []
        # for m in movie_test:
        #     X_test.append(uim.loc[:,m].values.tolist())
        
        # votes = pd.Series(index = movie_test,data = y_test)
        
        # #recuperer la prediction
        
        # pred = SVM(X_train, y_train, X_test, y_test, votes)
        
        user_test = test.loc[test.userId == u]
        movie_test = user_test.movieId.values
        y_test = user_test.apply(func = lambda x: 1 if x['rating'] >= 3.5 else 0,
                                   axis = 1).values
        
        user_train = train.loc[train.userId == u]
        movie_train = user_train.movieId.values

        y_train = user_train.apply(func = lambda x: 1 if x['rating'] >= 3.5 else 0, axis = 1)
            
        X_test = []
        for m in movie_test:
            X_test.append(uim.loc[:,m].values)
        X_train = []
        for m in movie_train:
            X_train.append(uim.loc[:,m].values.tolist())
            
        votes = pd.Series(index = movie_test,data = y_test)
        #si une seule classe 
        if len(y_train.unique()) != 2:
            seul += 1
            continue
        
        pred = SVM(X_train, y_train, X_test, votes,kernel)
        taille += 1   
        
        y = pred.values
        
        moy_accuracy += accuracy_score(y_test,y)
        
        for k in liste_k:
            p1,r1,p0,r0 = evaluation(votes,pred,k)
            #on fait la somme pour chaque k
            moy_precision_rel[k] += p1
            moy_recall_rel[k] += r1
            moy_precision_n_rel[k] += p0
            moy_recall_n_rel[k] += r0
            
        print("nb user : ",i)
        i += 1
        print("id user : ",u)
        
    taille = len(users)
    #accuracy
    print("une seule classe: ",seul)
    print("taille : ",taille)
    moy_accuracy /= taille
    print("mean accuracy : ",moy_accuracy)
    print("\n\n######  moyenne evaluation  #####\n")
    for k in liste_k:
        moy_precision_rel[k] /= taille
        moy_recall_rel[k] /= taille
        moy_precision_n_rel[k] /= taille
        moy_recall_n_rel[k] /= taille 
        
        
        #enregistrement
        
    moy_p = {1:moy_precision_rel,0:moy_precision_n_rel}
    moy_r = {1:moy_recall_rel,0:moy_recall_n_rel}  
    name = "SVM_new_"+kernel+"_moy_prec.pkl"
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_p)
        
    name = "SVM_new_"+kernel+"_moy_rec.pkl"    
    with open(name,'wb') as f:
        p = pickle.Pickler(f)
        p.dump(moy_r)  
     
    
    return moy_p,moy_r,moy_accuracy
        
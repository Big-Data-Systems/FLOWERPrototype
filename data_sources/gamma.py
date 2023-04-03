#!/usr/bin/env python
# coding: utf-8

# In[563]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# # Incremental PCA

# In[564]:


def compute_PCA(gamma):
    n = gamma[0,0]
    d = len(gamma)
    L = gamma[1:d,0] #[row,column]
    Q = gamma[1:d,1:d]

    corr_matrix = np.zeros([len(Q),len(Q)])

    for i in range(len(Q)):
        denom_1 = math.sqrt((n*Q[i,i]) - (L[i]*L[i]))
        for j in range(len(Q)):
            numerator = ((n*Q[i,j]) - (L[i]*L[j]))
            denom_2 = math.sqrt((n*Q[j,j]) - (L[j]*L[j]))
            corr_matrix[i,j] = numerator/(denom_1*denom_2)
           
    u, s, vh = np.linalg.svd(corr_matrix)
   
    return [u,s,vh]


# In[565]:


def compute_gamma(X):
    #print(X.shape)
    ones = [1]*X.shape[1]
    Z = np.vstack([ones,X])
    gamma = np.dot(Z,np.transpose(Z))
   
   
    return gamma


# In[566]:


iteration = 0
ChunkSize = 500

s_prev = float("inf")

for df in pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_05_values_spikes.csv", chunksize=ChunkSize):
    X = df.to_numpy() #a nXd numpy array
   
    # compute gamma
    if iteration == 0:
        gamma = compute_gamma(np.transpose(X))
    else:
        partial_gamma = compute_gamma(np.transpose(X))
        gamma += partial_gamma
    iteration +=1
       
    # compute PCA
    pca_model = compute_PCA(gamma)    
    u = pca_model[0]
    s = pca_model[1]
    vh = pca_model[2]
   
    #incremental: stopping earlier
    #if iteration>12 and abs(max(s)-s_prev)<0.001:
     #   break
       # initially smaller chunks 500, later 5000
    #s_prev = max(s)
       
   
   
#print("u= ", u)
print("Total iteration:", iteration)
print("Eigen values (s)= ",s)
#print("vh=",vh)


# In[567]:


df_u= pd.DataFrame(u)
df_u.to_csv("/Users/tahsin/Documents/Python_programs/u_05.csv",index=False)


# In[572]:


## get number of components to be selected (k)
variance_captured_so_far = []
k = 0 #top k components
for i in range(len(s)):
    variance_captured_so_far.append(sum(s[:i+1])/sum(s))
    print("Component selected:" +str(i) + ", variance_captured_so_far:"+str(variance_captured_so_far[-1]))
    #stop the loop at 90%
    if variance_captured_so_far[-1]>0.9:
        k = i
        break


# In[573]:


## select k columns (components) from vh
print("Before vh dimension:",vh.shape)
vh_k = vh[:, :k+1]
print("After vh dimension:",vh_k.shape)
#for i in range(len(vh)):
   


# In[574]:


##get the final matrix with K-dimensions
df_original = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_tranposed.csv")
X_k = np.dot(df_original.to_numpy(),vh_k)
print("new dimension of the dataset:",X_k.shape)
# to dataframe
df_k = pd.DataFrame(X_k)


# In[575]:


df_k.head(5)


# In[9]:


df_k.to_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_reducted.csv",index=False)


# # Incremental K-means

# In[10]:


import numpy as np
def compute_kmeans(gamma):
    N = []
    L = []
    Q = []
    prior = []
    mu = []
    sigma = []
    gamma = np.transpose(gamma)
   
    Nglobal = 0 #total number of rows
    for i in range(2):
        Nglobal+=gamma[0,i*2]    
    NumDim = np.shape(gamma)[0]-2 #get total dimensions

    #print(Nglobal, NumDim)
    print(gamma)
    for index in range(2):
        N.append(gamma[0,index*2])
        L.append(gamma[1:np.shape(gamma)[0],index*2])
        Q.append(gamma[1:np.shape(gamma)[0],index*2+1])
       
        #print(Q,L)
       
        print("Q and L: ")
        print(Q[index][0:NumDim],L[index][0:NumDim])
        mu.append(L[index][0:NumDim]/N[index]) # C
        sigma.append((Q[index][0:NumDim]/N[index]) - (L[index][0:NumDim]/N[index]**2)) #R
        prior.append(N[index]/Nglobal) # W
       
       

    return [mu, sigma, prior] #CRW


# In[11]:


def compute_k_gamma(X, k):
    n = X.shape[0]
    d = X.shape[1]
    class_col = d
   
    X_array = X.to_numpy()
    x = np.zeros(d+1)
    G = np.zeros((k*2,d+1))
    for i in range(0,n):
        c = int(X_array[i,class_col-1]) #getting class number
        #print(c)
        x[0] = 1
        for j in range(0,d):
            x[j+1] = X_array[i,j] #X with extra 1s
       
        for j in range(0,d+1):
            G[c*2,j] += x[j] * x[0]
            if j!=0:
                G[c*2+1,j] += x[j] * x[j]
    return G


# In[12]:


def assignCluster(X, cluster_centroids, k):
    cluster_id = [0]*len(X)
    sum_of_error = 0.0
    for i in range (0,len(X)):
        sq_diff_sum = [0]*k
        for j in range(k):
            clu_cent_vector = cluster_centroids[j]
            cluster_centroid_j = clu_cent_vector
            sq_diff_sum[j] = sum((X[i]-cluster_centroid_j)**2) ## X(i,_) -> ?
       
        cluster_id[i] = sq_diff_sum.index(min(sq_diff_sum)) #argmin
        sum_of_error += min(sq_diff_sum)
    ret = [cluster_id,sum_of_error]
    return ret


# In[13]:


import random
iteration = 0
ChunkSize = 1000
k = 2
for df in pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_reducted.csv",header = None, chunksize=ChunkSize):
    print(df)
    df = df.drop(4,axis=1) ##only for this practice dataset
   
    sum_of_error = 999999999
    prev_sum_of_error = -99999999
    X = df.to_numpy()
    onepass = 1
    cluster_centroids = []
    while True:
        print("\n### Iteration: "+str(iteration)+" ######")
        #compute cluster centroids randomly only for the first time and for first chunk
        if onepass==1:
            centroids = []
            for j in range(k):
                tmp = random.randint(0,len(X)-1)
                if tmp not in centroids:
                    centroids.append(tmp)
                    cluster_centroids.append(X[tmp])
                else:
                    j = j-1
        #print("cluster_centroids:",cluster_centroids)
        #get cluster numbers
        ret = assignCluster(X, cluster_centroids, len(cluster_centroids)) #assign the cluster number to each row in dataset
        #print("returned from assigncluster():",ret)
        cluster_id = ret[0] #get the cluster number list for all rows in the dataset
        sum_of_error =  ret[1] #get sum of error value to break out of loop when it reaches a constant value
        df["Cluster"] = cluster_id #append cluster_ids to the original data in order to send it to construct gamma matrix
        #print(df)
       

        if iteration == 0:
            k_gamma = compute_k_gamma(df,k)
        else:
            partial_k_gamma = compute_k_gamma(df,k)
            k_gamma += partial_k_gamma
           
           
        CRW = compute_kmeans(k_gamma)
        C = CRW[0]
       
        #print("Centroids from model: ")
        #print("C=",C)
        #print("R=",CRW[1])
        #print("W=",CRW[2])
       
        onepass+=1
        if math.sqrt(abs(prev_sum_of_error - sum_of_error))< 1:
            break

        cluster_centroids = C
        prev_sum_of_error = sum_of_error
        iteration +=1


# In[14]:


df


# # K-means with Selected components

# In[26]:


## Cluster visualization
df_cluster_vis = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_reducted.csv")


# In[576]:


df_cluster_vis = df_k
df_cluster_vis.shape


# In[577]:


#Import required module
from sklearn.cluster import KMeans
 
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_cluster_vis)
    kmeanModel.fit(df_cluster_vis)
    inertia.append(kmeanModel.inertia_)
   
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
   
#df_cluster_vis["label"] = pd.DataFrame(label)
#print(label)


# In[578]:


kmeans = KMeans(n_clusters=6).fit(df_cluster_vis)
df_cluster_vis["label"] = pd.DataFrame(kmeans.labels_)


# In[579]:


df_cluster_vis.columns = ["0","1","label"]


# In[580]:


pd.plotting.parallel_coordinates(df_cluster_vis, 'label', color=('#556270', '#4ECDC4', '#C7F464', '#0F0F0C'))


# In[581]:


import seaborn as sns
sns.lmplot(x="0",y="1",data=df_cluster_vis,hue='label',fit_reg=False)


# In[583]:


import plotly.express as px
fig = px.scatter(df_cluster_vis, x='0', y='1',color='label')
fig.show()


# # K-means with selected variables

# In[377]:


df_var_k = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_tranposed.csv")


# In[350]:


fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df_var_k.hist(ax=ax)


# In[378]:


df_var_k12 = df_var_k[["16","24"]]
df_var_k12.shape


# In[361]:


#Import required module
from sklearn.cluster import KMeans
 
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_var_k12)
    kmeanModel.fit(df_var_k12)
    inertia.append(kmeanModel.inertia_)
   
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
   
#df_cluster_vis["label"] = pd.DataFrame(label)
#print(label)


# In[379]:


kmeans = KMeans(n_clusters=4).fit(df_var_k12)
df_var_k12["label"] = pd.DataFrame(kmeans.labels_)


# In[380]:


import seaborn as sns
sns.lmplot(x="16",y="24",data=df_var_k12,hue='label',fit_reg=False)


# In[364]:


pd.plotting.parallel_coordinates(df_var_k12, 'label', color=('#556270', '#4ECDC4', '#C7F464', '#0F0F0C','#0000FF','#FAEBD7'))


# In[348]:


import plotly.express as px
fig = px.scatter_3d(df_var_k12, x='2', y='5', z='6',
              color='label')
fig.show()


# In[94]:


kmeans.cluster_centers_


# In[95]:


df_var_k12["label"].value_counts()


# In[96]:


df_var_k12.to_csv("/Users/tahsin/Documents/Python_programs/Phe02_spikes_k4_d4.csv",index=False)


# In[97]:


df_var_k12.hist()


# # Channel: 3 (Phe_03): K-means

# In[435]:


df_var_k = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_03_values_spikes.csv")


# In[436]:


df_var_k12 = df_var_k[["16","24"]]
df_var_k12.shape


# In[437]:


#Import required module
from sklearn.cluster import KMeans
 
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_var_k12)
    kmeanModel.fit(df_var_k12)
    inertia.append(kmeanModel.inertia_)
   
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
   
#df_cluster_vis["label"] = pd.DataFrame(label)
#print(label)


# In[438]:


kmeans = KMeans(n_clusters=7).fit(df_var_k12)
df_var_k12["label"] = pd.DataFrame(kmeans.labels_)


# In[439]:


import seaborn as sns
sns.lmplot(x="16",y="24",data=df_var_k12,hue='label',fit_reg=False)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")
plt.show()


# In[200]:


pd.plotting.parallel_coordinates(df_var_k12, 'label', color=('#556270', '#4ECDC4', '#C7F464', '#0F0F0C','#0000FF','#FAEBD7'))


# In[441]:


import plotly.express as px
#fig = px.scatter_3d(df_var_k12, x='6', y='16', z='24',
              #color='label')
fig = px.scatter(df_var_k12, x='16', y='24',
              color='label')
fig.show()
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")
#plt.show()


# # Channel: 4 (Phe_04): K-means

# In[512]:


df_var_k = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_04_values_spikes.csv")


# In[513]:


df_var_k12 = df_var_k[["16"]]
df_var_k12.shape


# In[ ]:


#Import required module
from sklearn.cluster import KMeans
 
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_var_k12)
    kmeanModel.fit(df_var_k12)
    inertia.append(kmeanModel.inertia_)
   
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
   
#df_cluster_vis["label"] = pd.DataFrame(label)
#print(label)


# In[ ]:


kmeans = KMeans(n_clusters=6).fit(df_var_k12)
df_var_k12["label"] = pd.DataFrame(kmeans.labels_)


# In[ ]:


import seaborn as sns
sns.lmplot(x="16",y="24",data=df_var_k12,hue='label',fit_reg=False)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")
plt.show()


# In[302]:


pd.plotting.parallel_coordinates(df_var_k12, 'label', color=('#556270', '#4ECDC4', '#C7F464', '#0F0F0C','#0000FF','#FAEBD7'))


# In[518]:


import plotly.express as px
#fig = px.scatter_3d(df_var_k12, x='2', y='16', z='24',
              #color='label')
fig = px.scatter(df_var_k12, x='16',
              color='label')
fig.show()


# # Channel: 5 (Phe_05): K-means

# In[442]:


df_var_k = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_05_values_spikes.csv")


# In[443]:


df_var_k12 = df_var_k[["16","24"]]
df_var_k12.shape


# In[373]:


#Import required module
from sklearn.cluster import KMeans
 
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_var_k12)
    kmeanModel.fit(df_var_k12)
    inertia.append(kmeanModel.inertia_)
   
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
   
#df_cluster_vis["label"] = pd.DataFrame(label)
#print(label)


# In[444]:


kmeans = KMeans(n_clusters=2).fit(df_var_k12)
df_var_k12["label"] = pd.DataFrame(kmeans.labels_)


# In[446]:


import seaborn as sns
sns.lmplot(x="16",y="24",data=df_var_k12,hue='label',fit_reg=False)
#plt.plot(kmeans.cluster_centers_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")
plt.show()


# In[409]:





# In[337]:


pd.plotting.parallel_coordinates(df_var_k12, 'label', color=('#556270', '#4ECDC4', '#C7F464', '#0F0F0C','#0000FF','#FAEBD7'))


# In[445]:


import plotly.express as px
#fig = px.scatter_3d(df_var_k12, x='6', y='16', z='24',
              #color='label')
fig = px.scatter(df_var_k12, x='16', y='24',
              color='label')
fig.show()


# In[545]:


#print(len(df["16"]))
df_var_k = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_04_values_spikes.csv")
#df_var_k = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_reducted.csv")
df_var_k12 = df_var_k[["16"]]
kmeans = KMeans(n_clusters=4).fit(df_var_k12)
df_var_k["label"] = pd.DataFrame(kmeans.labels_)

print(len(df_var_k12))
#df_window1 = df_var_k[(df_var_k12["16"]>0.00) & (df_var_k["16"]<0.07)]
print(len(df_window1))
"""
df_window = df_window1[(df_window1["24"]>-0.03)&(df_window1["24"]<0.02)]
#df_window2 = df_var_k12[(df_var_k12["24"]>-0.03)&(df_var_k12["24"]<0.02)]
print(df_window.size)
plt.scatter(df_window["16"],df_window["24"])
"""


# In[ ]:





# In[546]:


import seaborn as sns
#sns.set(rc={'figure.figsize':(26.7,15.27)})
sns.lmplot(x="16",y="24",data=df_var_k,hue='label',fit_reg=False)
#plt.plot(df_var_k12["label"],df_var_k12["16"])
#plt.plot(kmeans.cluster_centers_)
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")
plt.show()


# # Correaltion between channels

# In[556]:


df_ch2 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_tranposed.csv")
df_ch3 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_03_values_spikes.csv")

corr23 = df_ch2.corrwith(df_ch3)
#print(corr23)
print(corr23[abs(corr23)>0.3])
print(max(corr23),min(corr23))


# In[557]:


df_ch2 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_tranposed.csv")
df_ch4 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_04_values_spikes.csv")

corr24 = df_ch2.corrwith(df_ch4)
#print(corr23)
print(corr24[abs(corr24)>0.3])
print(max(corr24),min(corr24))


# In[558]:


df_ch2 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_02_values_spikes_windowsresults_tranposed.csv")
df_ch5 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_05_values_spikes.csv")

corr25 = df_ch2.corrwith(df_ch5)
#print(corr25)
print(corr25[abs(corr25)>0.3])
print(max(corr25),min(corr25))


# In[560]:


df_ch3 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_03_values_spikes.csv")
df_ch4 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_04_values_spikes.csv")

corr34 = df_ch3.corrwith(df_ch4)
#print(corr23)
print(corr34[abs(corr34)>0.3])
print(max(corr34),min(corr34))


# In[561]:


df_ch3 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_03_values_spikes.csv")
df_ch5 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_05_values_spikes.csv")

corr35 = df_ch3.corrwith(df_ch5)
#print(corr23)
print(corr35[abs(corr35)>0.3])
print(max(corr35),min(corr35))


# In[562]:


df_ch4 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_04_values_spikes.csv")
df_ch5 = pd.read_csv("/Users/tahsin/Documents/Python_programs/Phe2_05_values_spikes.csv")

corr45 = df_ch4.corrwith(df_ch5)
#print(corr23)
print(corr45[abs(corr45)>0.3])
print(max(corr45),min(corr45))


# In[ ]:
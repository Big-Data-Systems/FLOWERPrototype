from clustering.dbscan import dbscan
from clustering.MAE_RMSE import MAE_RMSE
from sklearn.externals import joblib
from dataset import DataSet
from preprocess import mean
import numpy as np


data = DataSet(
    '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
    '/home/imad/Desktop/PFE/db/ml-100k/u.user',
    '/home/imad/Desktop/PFE/db/ml-100k/u.item'
)

print("#FC user user | mean user | k-medoids")
usage_matrix = mean(data.get_usage_matrix())
distance_matrix = joblib.load('./distance_matrices/fc_uu_ua_dist_mat.sav')

dbs = dbscan(
    distance_matrix,
    0.4,
    50,
    ccore=False,
    data_type="distance_matrix"
)
dbs.process()
print(dbs.get_clusters())
print(len(dbs.get_clusters()), len(dbs.get_clusters()[0]))
#print(kmed.get_medoids())

print(MAE_RMSE(usage_matrix, distance_matrix, dbs.get_clusters(), '/home/imad/Desktop/PFE/db/ml-100k/ua.test'))
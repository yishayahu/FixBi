import pickle
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

lbls = pickle.load(open('labels_in_cluster_source.p','rb'))
lblt = pickle.load(open('labels_in_cluster_target.p','rb'))
x=4
inds = linear_sum_assignment(lblt,maximize=True)
print(np.sum(lblt[inds]) / np.sum(lblt))
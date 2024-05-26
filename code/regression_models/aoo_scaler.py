#!/usr/bin/env python3

### Standard scaler for AOO values ###

# Dependencies
from sklearn.preprocessing import StandardScaler
import os
import pickle
import numpy as np

# Working directory
os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')

# Data directory
data_dir = "data/features/"

#  Load outcome vector
y = np.loadtxt(data_dir + "aoo.txt", delimiter='\t', usecols=[1], skiprows=1)

# Fit scaler to default range (0,1)
aooScaler = StandardScaler().fit(y.reshape(-1, 1))

# Pickle scaler
with open(data_dir + 'aooStandardScaler.pkl', 'wb') as f:
    pickle.dump(aooScaler, f)
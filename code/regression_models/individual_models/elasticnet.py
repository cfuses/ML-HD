#!/usr/bin/env python3

### Elastic Net ###

# Dependencies
from sklearn import linear_model, model_selection
import numpy as np
import os 
import pickle

# Load custom functions
from evaluating_functions import model_description, model_metrics
from data_loading import read_sparse_X, scale_CAG

# Printing time for log recording
from datetime import datetime
def _print(*args, **kw):
    print("[%s]" % (datetime.now()),*args, **kw)
    
_print("Start time")

#--------# Directories #--------#

# Change working directory
os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')
# os.chdir('/gpfs/projects/bsc83/MN4/bsc83/Projects/Creatio/enroll_hd/')

# Data directory
data_dir = "data/features/"

# Input files
X_path = data_dir + "X_pc10_filt_0.01.txt"
y_path = data_dir + "y_pc10.txt"
# X_path = data_dir + "feature_matrix_m3_filt_0.01_nodups.txt"
# y_path = data_dir + "aoo.txt"

# Results directory
results_dir = "data/ml_results/"

#--------# Load data #--------#

# Load X matrix
X = read_sparse_X(X_path, chunk_size = 100)

# Load outcome vector
y = np.loadtxt(y_path, delimiter='\t', usecols=[1], skiprows=1)

_print("Data loaded.")

#--------# Scale CAG and AOO #--------#

# Scale CAG column
X = scale_CAG(X)

# Import AOO scaler
with open(data_dir + 'aooStandardScaler.pkl', 'rb') as f:
    aooScaler = pickle.load(f)

# Scale outcome vector (AOO)
y = aooScaler.transform(y.reshape(-1, 1))

#--------# Split train and test #--------#

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 51)

# Convert aoo to 1D array
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#--------# Model #--------#

# Define model
elastic_net = linear_model.ElasticNet(max_iter=10000)

# Define the parameter grid to search
param_grid = {
    'alpha': [0.005, 0.01, 0.05],  # Regularization strength
    'l1_ratio': [0.2, 0.5, 0.9],  # Mixing parameter for L1 (0) and L2 (1) penalties
}
# Perform GridSearchCV with cross-validation
elastic_net_grid_search = model_selection.GridSearchCV(elastic_net, param_grid, cv=5)
elastic_net_grid_search.fit(X_train, y_train)

# Extract best hyperparameter
elastic_net_best = elastic_net_grid_search.best_estimator_

_print("Best trained estimator:", elastic_net_best)
_print("Saving...")

# See how the model has trained
evalplot = model_description(elastic_net_best, X_train, y_train, scaler = aooScaler)
evalplot.savefig(results_dir + 'elastic_net_training.png')

# Compute model metrics
met, params, errorplot = model_metrics(elastic_net_best, X_test, y_test, scaler = aooScaler)
with open(results_dir + 'elastic_net_metrics.txt', 'w') as file:
    file.write(met)
    file.write(params)
errorplot.savefig(results_dir + 'elastic_net_prediction.png')

# Pickle best estimator
with open(results_dir + 'regressors/elastic_net_regressor.pkl', 'wb') as f:
    pickle.dump(elastic_net_best, f)
    
_print("Results saved.")
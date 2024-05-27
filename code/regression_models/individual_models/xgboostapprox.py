#!/usr/bin/env python3

### XGBoost using approx tree method ###

# Dependencies
from xgboost import XGBRegressor
import numpy as np
import os 
import pickle
from sklearn import model_selection

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
# os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')
os.chdir('/gpfs/projects/bsc83/MN4/bsc83/Projects/Creatio/enroll_hd/')

# Data directory
data_dir = "data/features/"

# Input files
# X_path = data_dir + "X_pc10_filt_0.01.txt"
# y_path = data_dir + "y_pc10.txt"
X_path = data_dir + "feature_matrix_m3_filt_0.01.txt"
y_path = data_dir + "aoo.txt"

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
XGBoostReg = XGBRegressor(tree_method="approx")

# Hyperparameters to tune
param_grid = {
    'max_depth': [3, 5],
    'reg_alpha': [0, 0.1, 0.5, 0.8],
    'n_estimators': [10, 20, 40, 80]
}
# Define grid search
XGBoostReg_grid_search = model_selection.GridSearchCV(estimator=XGBoostReg, 
                                                      param_grid=param_grid, 
                                                      scoring='r2', cv=5, 
                                                      n_jobs=-1, verbose=3)

# Train model
XGBoostReg_grid_search.fit(X_train, y_train)

# Extract best model
XGBoostRegBest = XGBoostReg_grid_search.best_estimator_

print(XGBoostReg_grid_search.best_params_)

_print("Estimator trained.")

# See how the model has trained
evalplot = model_description(XGBoostRegBest, X_train, y_train, scaler = aooScaler)
evalplot.savefig(results_dir + 'approxXGBoost_training.png')

# Compute model metrics
met, params, errorplot = model_metrics(XGBoostRegBest, X_test, y_test, scaler = aooScaler)
with open(results_dir + 'approxXGBoost_metrics.txt', 'w') as file:
    file.write(met)
    file.write(params)
errorplot.savefig(results_dir + 'approxXGBoost_prediction.png')

# Pickle best estimator
with open(results_dir + 'regressors/approxXGBoost_regressor.pkl', 'wb') as f:
    pickle.dump(XGBoostRegBest, f)
    
_print("Results saved.")
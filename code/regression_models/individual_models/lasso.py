#!/usr/bin/env python3

### Lasso regression ###

# Dependencies
from sklearn import linear_model, model_selection
import numpy as np
import os 
import pickle

# Load custom functions
from evaluating_functions import model_description, model_metrics
from data_loading import read_sparse_X, scale_CAG

#--------# Directories #--------#

# Change working directory
# os.chdir('/media/HDD_4TB_1/jordi/cfuses_gnn_enrollhd_2024/')
os.chdir('/gpfs/projects/bsc83/MN4/bsc83/Projects/Creatio/enroll_hd/')

# Data directory
data_dir = "data/features/"

# Input files
# X_path = data_dir + "X_pc10_filt_0.01.txt"
# y_path = data_dir + "y_pc10.txt"
X_path = data_dir + "feature_matrix_m3_filt_0.01_nodups.txt"
y_path = data_dir + "aoo.txt"

# Results directory
results_dir = "data/ml_results/"

#--------# Load data #--------#

# Load X matrix
X = read_sparse_X(X_path, chunk_size = 100)

print("X loaded, taking", (X.dtype.itemsize * X.size)/1e6,'MB')

# Load outcome vector
y = np.loadtxt(y_path, delimiter='\t', usecols=[1], skiprows=1)

print("Data loaded.")

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
lasso = linear_model.Lasso()

# Define the parameter grid to search
param_grid = {
    'alpha': [0.005, 0.01, 0.05],  # Regularization strength
    'max_iter': [1000, 2000]  # Maximum number of iterations
}
# Perform GridSearchCV with cross-validation
lasso_grid_search = model_selection.GridSearchCV(lasso, param_grid, cv=5)
lasso_grid_search.fit(X_train, y_train)

# Extract best hyperparameter
lasso_best = lasso_grid_search.best_estimator_

print("Best trained estimator:", lasso_best)
print("Saving...")

# See how the model has trained
evalplot = model_description(lasso_best, X_train, y_train, scaler = aooScaler)
evalplot.savefig(results_dir + 'lasso_training.png')

# Compute model metrics
met, params, errorplot = model_metrics(lasso_best, X_test, y_test, scaler = aooScaler)
with open(results_dir + 'lasso_metrics.txt', 'w') as file:
    file.write(met)
    file.write(params)
errorplot.savefig(results_dir + 'lasso_prediction.png')

# Pickle best estimator
with open(results_dir + 'regressors/lasso_regressor.pkl', 'wb') as f:
    pickle.dump(lasso_best, f)
    
print("Results saved.")
#!/usr/bin/env python3

### OLS using sex and cag as regressors ###

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

# Take selected regressors columns
X_train_noSNPs = X_train[:,:2]
X_test_noSNPs = X_test[:,:2]

# Convert aoo to 1D array
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#--------# Model #--------#

# Define model
ols = linear_model.LinearRegression()

_print("Model fit start.")

# Perform cross-validation with scoring metric
cv_results = model_selection.cross_validate(ols, X_train_noSNPs, y_train, cv=5, scoring='r2', return_estimator=True)

_print("Model fit end.")

# Get the index of the fold with the best performance
best_estimator_index = cv_results['test_score'].argmax()

# Retrieve the best estimator's hyperpamateres from the best fold (with higher score)
best_estimator = cv_results['estimator'][best_estimator_index]

# Train the final model using the best hyperparameter values
ols_best = best_estimator.fit(X_train_noSNPs, y_train)

_print("Best trained estimator:", ols_best)
_print("Saving...")

# See how the model has trained
evalplot = model_description(ols_best, X_train_noSNPs, y_train, scaler = aooScaler)
evalplot.savefig(results_dir + 'ols_sexcag_training.png')

# Compute model metrics
met, params, errorplot = model_metrics(ols_best, X_test_noSNPs, y_test, scaler = aooScaler)
with open(results_dir + 'ols_sexcag_metrics.txt', 'w') as file:
    file.write(met)
    file.write(params)
errorplot.savefig(results_dir + 'ols_sexcag_prediction.png')

# Pickle best estimator
with open(results_dir + 'regressors/ols_sexcag.pkl', 'wb') as f:
    pickle.dump(ols_best, f)
    
_print("Results saved.")
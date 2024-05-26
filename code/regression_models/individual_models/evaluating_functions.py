#!/usr/bin/env python3

### Functions for model evaluation ###

# Dependencies
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


def model_description(regressor, X_train, y_train, scaler=None):
    '''Describes regressor by plotting actual vs predicted 
    values of the train set. Percentage of deviance explained on plot title.
    
    Actual vs predicted values are transformed to their original range if
    the transformation scaler is passed as argument.
    '''
    # Get aoo predicted
    y_train_predicted = regressor.predict(X_train)

    # Compute D^2 (percentage of deviance explained) of train set
    dev_explained = metrics.explained_variance_score(y_true = y_train, y_pred = y_train_predicted)

    # inverse transform of outcome vector
    if scaler != None:
        y_train_predicted = scaler.inverse_transform(y_train_predicted.reshape(-1, 1)).reshape(-1, 1).astype(np.float32)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1, 1).astype(np.float32)

    # Plot prediction versus actual values
    evalplot, ax = plt.subplots()
    
    # Create a heatmap
    hb = ax.hexbin(y_train_predicted, y_train, cmap='Oranges', gridsize=15,mincnt=1,bins='log')

    # Plot the diagonal line
    ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")

    # Set x limits to span from minimum to maximum of the data
    ax.set_xlim(y_train_predicted.min()-0.1*y_train_predicted.min(), 
                y_train_predicted.max()+0.1*y_train_predicted.max())
    
    # Add a colorbar
    cb = evalplot.colorbar(hb, ax=ax)
    cb.set_label('Counts')
    
    # Set labels and title
    ax.set_ylabel('Actual values')
    ax.set_xlabel('Predicted values')
    
    evalplot.suptitle('Actual vs. Predicted values of train set by ' + str(regressor).split('(')[0])
    ax.set_title('% of deviance explained: ' + str(round(dev_explained,4)))
    
    return evalplot
    
def model_metrics(regressor, X_test, y_test, scaler=None):
    '''Evaluates regressor predicting test samples and returning:
    - coefficient of determination (R^2) 
    - mean squared error (MSE)
    - mean absolute error (MAE)
    - Visual evaluation of:
        - true vs predicted AOO
        - residuals vs predicted
        
    Actual and predicted values are plotted transformed to their 
    original range if the transformation scaler is passed as argument.
    '''

    # Get aoo predicted
    y_test_predicted = regressor.predict(X_test)

    # Compute R^2 (Coefficient of determination)
    R2 = metrics.r2_score(y_test, y_test_predicted)

    # Compute MSE (Mean Squared Error)
    MSE = metrics.mean_squared_error(y_test, y_test_predicted)

    # Compute MAE (Mean Absolute Error)
    MAE = metrics.mean_absolute_error(y_test, y_test_predicted)

    # Save metrics in list
    mlist = [R2, MSE, MAE]

    # Convert the metrics and their values into a string
    metrics_string = 'Regression model {} metrics:\n'.format(str(regressor).split('(')[0])
    for metric, value in zip(['R^2', 'MSE', 'MAE'], mlist):
        metrics_string += '{}: {}\n'.format(metric, round(value, 4))
    
    # Convert regressor parameters into string
    params_string = '\nRegression model parameters:\n' 
    params = regressor.get_params()
    for key, value in params.items():
        params_string += f"{key}: {value}\n"
        
    # inverse transform minmax scaling of outcome vector
    if scaler != None:
        y_test_predicted = scaler.inverse_transform(y_test_predicted.reshape(-1, 1)).reshape(-1, 1).astype(np.float32)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, 1).astype(np.float32)
    
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
   
    # Actual vs predicted
    hb0 = axs[0].hexbin(y_test_predicted, y_test, cmap='Oranges', gridsize=15,mincnt=1,bins='log')

    # Plot the diagonal line
    axs[0].plot(axs[0].get_ylim(), axs[0].get_ylim(), ls="--", c=".3")    
    
    # # Set x limits to span from minimum to maximum of the data
    axs[0].set_xlim(y_test_predicted.min()-0.1*y_test_predicted.min(), 
                    y_test_predicted.max()+0.1*y_test_predicted.max())
    
    # Set labels and title
    axs[0].set_ylabel('Actual values')
    axs[0].set_xlabel('Predicted values')
    axs[0].set_title("Actual vs. Predicted values")
    
    # Residuals vs predicted
    # Create a heatmap
    hb1 = axs[1].hexbin(y_test_predicted, y_test-y_test_predicted, cmap='Oranges', gridsize=15, mincnt=1, bins='log')

    # Plot the straight line
    axs[1].axhline(y=0, ls="--", c=".3")

    # Set labels and title
    axs[1].set_ylabel('Residuals (actual - predicted)')
    axs[1].set_xlabel('Predicted values')
    axs[1].set_title("Residuals vs. Predicted Values")    
    
    # Create an axis for the colorbar
    cax = fig.add_axes([0.9, 0.1, 0.02, 0.75])  # [left, bottom, width, height]

    # Add a colorbar
    cb = fig.colorbar(hb1, cax=cax)
    cb.set_label('Counts')
    
    fig.suptitle(str(regressor).split('(')[0] + ' (r2 = {})'.format(str(round(R2,3))))
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.88, top=0.85, bottom=0.13, wspace=0.3, hspace=0.3)
    
    # Return metrics list and plot
    return metrics_string, params_string, fig
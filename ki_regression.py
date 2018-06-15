import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import matplotlib.patches as mpatches

ligand_data = pickle.load(open('Ligand_data_clean.pickle', 'rb'))

ligand_features = ligand_data['ligand_features']
ligand_props = ligand_data['ligand_props']


def save_resid_plot(model, x_train, y_train, x_test, y_test, title, file_name):
    """
    Saves scatter plot of residuals vs. predicted values for training and test sets
    for a given model
    :param model:sklearn linear_model trained on x_train & y_train
    :param x_train: Training features
    :param y_train: Training outcomes
    :param x_test: Test features
    :param y_test: Test outcomes
    :param title: Title for plot
    :param file_name: file name to save .png
    :return: None
    """
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    plt.scatter(y_pred_train, y_pred_train - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(y_pred_test, y_pred_test - y_test, c='r', s=40, alpha=0.5)
    plt.hlines(y=0, xmin=min(min(y_pred_train), min(y_pred_test)), xmax=max(max(y_pred_train), max(y_pred_test)))
    plt.ylabel('Residuals')
    plt.xlabel('Predicted log Ki')
    plt.title(title)
    handles = [mpatches.Patch(color='blue', label='Training'),
               mpatches.Patch(color='red', label='Test')]
    plt.legend(handles=handles)
    plt.savefig(file_name)
    plt.gcf().clear()

    return


def range_std(x):
    """
    Scales a vector x to be within range [0,1]
    :param x: numeric vector
    :return: scaled numeric vector
    """
    return (x-min(x))/(max(x)-min(x))


# Truncated Ki 10000K values - plot results with and without?
ligand_props = ligand_props.loc[~(ligand_props['ki.Val'] == 10000)]
ligand_features = ligand_features.loc[ligand_props.index]

# Centre features and outcome to be within range 0-1
ligand_features = ligand_features.apply(axis=0, func=range_std)
ligand_props['log.ki'] = range_std(ligand_props['log.ki'])

# Generate train/test split
feature_train, feature_test, props_train, props_test = model_selection.train_test_split(
    ligand_features, ligand_props, test_size=0.2, random_state=0)

# OLS Regression
lin_model = linear_model.LinearRegression(fit_intercept=True)
lin_model.fit(X=feature_train, y=props_train['log.ki'])
train_pred = lin_model.predict(feature_train)
test_pred = lin_model.predict(feature_test)
# Print regression metrics
print("OLS regression")
print("R-squared: ", lin_model.score(X=feature_train, y=props_train['log.ki']))
print("Test MSE: ", metrics.mean_squared_error(props_test['log.ki'], test_pred))
print("Train MSE: ", metrics.mean_squared_error(props_train['log.ki'], train_pred))
# Save residual plot for train & test set
save_resid_plot(
    lin_model, feature_train, props_train['log.ki'], feature_test, props_test['log.ki'],
    "OLS Regression\n Residuals vs. Predicted log Ki", "OLS_resids.png")


# Elastic net regularization
enet_model = linear_model.ElasticNetCV(cv=15)
enet_model.fit(X=feature_train, y=props_train['log.ki'])
test_pred = enet_model.predict(feature_test)
train_pred = enet_model.predict(feature_train)
print("\nElastic Net")
print("R-squared: ", enet_model.score(X=feature_train, y=props_train['log.ki']))
print("Test MSE: ", metrics.mean_squared_error(props_test['log.ki'], test_pred))
print("Train MSE: ", metrics.mean_squared_error(props_train['log.ki'], train_pred))
# Save residual plot for train & test set of enet
save_resid_plot(
    enet_model, feature_train, props_train['log.ki'], feature_test, props_test['log.ki'],
    "Regression w/ Elastic net Regularization \nResiduals vs. Predicted log Ki", "Enet_resids.png")


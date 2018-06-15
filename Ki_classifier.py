import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics

# Load pre-processed data
ligand_data = pickle.load(open('Ligand_data_clean.pickle', 'rb'))

ligand_features = ligand_data['ligand_features']
ligand_props = ligand_data['ligand_props']

# Create new binary variable 'binding' from Ki.Val
ligand_props['binding'] = ligand_props['ki.Val'] < 1000
# 154 non binding (Ki >= 1000)
# 234 binding (ki < 1000)

# Balancing class sizes
# Subset binding samples to 154 with lowest affinity - Don't want border cases
lowki_154_idx = ligand_props.sort_values(by='ki.Val',).index[0:154]
nonbinding_idx = ligand_props[ligand_props['binding'] == False].index
sub_idx = lowki_154_idx.append(nonbinding_idx)
feature_sub = ligand_features.loc[sub_idx]
props_sub = ligand_props.loc[sub_idx]


def save_roc_plot(classif, x_train, y_train, x_test, y_test, plot_title, file_name):
    """
    Saves a plot of ROC curves of a given classifier for both training and test sets
    :param classif: sklearn binary classifier which predict_proba() can be called on
    :param x_train: pd.DataFrame of Training features
    :param y_train: pd.Training outcomes
    :param x_test: Test features
    :param y_test: Test outcomes
    :param plot_title: Title of plot
    :param file_name: file name to save .png
    :return:
    """
    y_prob_test = classif.predict_proba(x_test)[::, 1]
    y_prob_train = classif.predict_proba(x_train)[::, 1]

    fpr_test, tpr_test, thresh_test = metrics.roc_curve(y_test, y_prob_test, drop_intermediate=False)
    fpr_train, tpr_train, thresh_train = metrics.roc_curve(y_train, y_prob_train, drop_intermediate=False)

    auc_test = metrics.roc_auc_score(y_test, y_prob_test)
    auc_train = metrics.roc_auc_score(y_train, y_prob_train)

    plt.plot(fpr_test, tpr_test, label='Test AUC=' + str(auc_test))
    plt.plot(fpr_train, tpr_train, label='Train AUC=' + str(auc_train))
    plt.legend(loc=4)
    plt.title(plot_title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(file_name)
    plt.gcf().clear()

    return


# Performance of logistic classifier over logit_iter train/test splits
logit_iter = 300
logit_perf_df = pd.DataFrame(index=np.arange(0, logit_iter), columns=['TN', 'FP', 'FN', 'TP'])
for boot_iter in range(0,logit_iter):

    feature_train, feature_test, props_train, props_test = model_selection.train_test_split(
        feature_sub, props_sub, test_size=0.2, stratify=props_sub['binding'])

    logit_class = linear_model.LogisticRegressionCV()
    logit_class.fit(feature_train, props_train['binding'])

    test_pred = logit_class.predict(feature_test)
    confus_mat = pd.crosstab(
        props_test['binding'], test_pred, rownames=['True binding'],
        colnames=['Predicted binding']).apply(lambda x: x/x.sum(), axis=1)
    confus_mat_flat = confus_mat.values.flatten()
    logit_perf_df.loc[boot_iter] = confus_mat_flat

print('Logistic classifier performance on test set')
print(logit_perf_df.apply(axis=0, func=np.mean))
print("Std Err.")
print(logit_perf_df.apply(axis=0, func=lambda x: np.std(x)/len(x)**0.5))
save_roc_plot(
    logit_class, feature_train, props_train['binding'], feature_test, props_test['binding'],
    'Logistic Classifier ROC', 'logistic_ROC.png')
logit_perf_df.to_csv("Logistic_performance.csv")

# Performance estimate of RF over rf_iter samplings of train/test split
rf_iter = 300
rf_perf_df = pd.DataFrame(index=np.arange(0, rf_iter), columns=['TN', 'FP', 'FN', 'TP'])
for boot_iter in range(0,rf_iter):

    feature_train, feature_test, props_train, props_test = model_selection.train_test_split(
        feature_sub, props_sub, test_size=0.2, stratify=props_sub['binding'])

    rf_class = RandomForestClassifier()
    rf_class.fit(feature_train, props_train['binding'])

    test_pred = rf_class.predict(feature_test)
    confus_mat = pd.crosstab(
        props_test['binding'], test_pred, rownames=['True binding'],
        colnames=['Predicted binding']).apply(lambda x: x/x.sum(), axis=1)
    confus_mat_flat = confus_mat.values.flatten()
    rf_perf_df.loc[boot_iter] = confus_mat_flat

print('RF classifier performance on test set')
print(rf_perf_df.apply(axis=0, func=np.mean))
print("Std Err.")
print(rf_perf_df.apply(axis=0, func=lambda x: np.std(x)/(len(x)**0.5)))
save_roc_plot(
    rf_class, feature_train, props_train['binding'], feature_test, props_test['binding'],
    'Random Forest Classifier ROC', 'randForest_ROC.png')
rf_perf_df.to_csv("RandomForest_performance.csv")


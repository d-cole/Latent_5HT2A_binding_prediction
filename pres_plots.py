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

plt.hist(ligand_props['ki.Val'],bins=15)
plt.xlabel("Ki")
plt.ylabel("Frequency")
plt.title("5-HT2A Ligand Ki Distribution")
plt.savefig("Ki_raw_dist.png")
plt.gcf().clear()

plt.hist(ligand_props['log.ki'],bins=15)
plt.xlabel("Log_e Ki")
plt.ylabel("Frequency")
plt.title("5-HT2A Ligand log Ki Distribution")
plt.savefig("Ki_log_dist.png")




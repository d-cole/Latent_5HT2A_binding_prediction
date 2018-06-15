import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# load 5-ht2a subset of KiDB
ht2a_subset = pd.read_csv('HT2A_kiDB.csv')

# subset to relevant columns
ht2a_subset = ht2a_subset[['Ligand.ID', 'Ligand.Name', 'SMILES', 'ki.Val']]

# Visualize Ki distribution
plt.hist(np.log(ht2a_subset['ki.Val'] + 1))
plt.savefig('ki_raw_dist.png')

# Take log of ki to get a more even distribution for IQR filtering
ht2a_subset['log.ki'] = np.log(ht2a_subset['ki.Val'] + 1)

# Handling replicate measures of ki
# For a ligand with replicate ki values the median of these values
#  is taken as the new ki as long as they all fall within a reasonable range.
dup_ligands = ht2a_subset.loc[ht2a_subset['Ligand.ID'].duplicated(keep=False)]
ndup_ligands = ht2a_subset.loc[~ht2a_subset['Ligand.ID'].duplicated(keep=False)]

# Get IQR of ligand log-ki values
iqr = np.percentile(ht2a_subset['log.ki'], 75) - np.percentile(ht2a_subset['log.ki'], 25)

for ligand_id in dup_ligands['Ligand.ID'].unique():
    ligand_rows = dup_ligands.loc[dup_ligands['Ligand.ID'] == ligand_id]
    ki_vals = ligand_rows['log.ki']
    ki_range = np.max(ki_vals) - np.min(ki_vals)

    # Experimental data is too variable - ligand is dropped
    if ki_range > iqr:
        pass

    # Experimental data isn't too variable - keep ligand and set ki to median of duplicates
    else:
        ligand_row = ligand_rows.groupby(['Ligand.ID', 'Ligand.Name', 'SMILES']).median().reset_index()
        ndup_ligands = pd.concat([ndup_ligands, ligand_row])

ndup_ligands = ndup_ligands.reset_index(drop=True)

# Handling Enantiomers -  a different kind of duplicate measure
# Enantiomers have same SMILE, but different Ligand.ID, Name, and possibly ki
# Handle enantiomers the same as duplicates, take the median(ki) and drop ligands if they are too variable
clean_ligands = ndup_ligands[~ndup_ligands['SMILES'].duplicated(keep=False)]
enantiomers = ndup_ligands.loc[ndup_ligands['SMILES'].duplicated(keep=False)]

for enan_smile in enantiomers['SMILES'].unique():
    enan_rows = enantiomers.loc[enantiomers['SMILES'] == enan_smile]
    ki_vals = enan_rows['log.ki']
    ki_range = np.max(ki_vals) - np.min(ki_vals)

    if ki_range > iqr:
        pass

    else:
        # Arbitrarily pick first enantiomer id and name
        enan_row = enan_rows.iloc[[0]]
        enan_row['ki.Val'] = np.median(enan_rows['ki.Val'])
        enan_row['log.ki'] = np.median(enan_rows['log.ki'])

        clean_ligands = pd.concat([clean_ligands, enan_row])

clean_ligands = clean_ligands.reset_index(drop=True)

# Subset clean_ligand data to the valid SMILEs returned from VAE
encoding_dict = pickle.load(open('smile_encodings.pickle', 'rb'))
encoded_smiles = encoding_dict['smile_encode']

# Clean up Series of numpy arrays to df
encoded_smiles_df = pd.DataFrame(encoded_smiles.apply(func=lambda x: pd.Series(x[0])))
smile_strings = encoding_dict['focal_smiles']
encoded_smiles_df = encoded_smiles_df.loc[smile_strings.SMILE.isin(clean_ligands.SMILES)]
smile_id_map = smile_strings.merge(clean_ligands[['Ligand.ID', 'SMILES']],
                                   left_on='SMILE', right_on='SMILES', how='inner')
# Set smiles_df index to be ligand ids
encoded_smiles_df = encoded_smiles_df.set_index(smile_id_map['Ligand.ID']).sort_index()

# Subset clean_ligands to ligands in encoded_df, set index to ligand_id and sort
clean_ligands = clean_ligands.loc[clean_ligands['Ligand.ID'].isin(encoded_smiles_df.index)]
clean_ligands = clean_ligands.set_index(['Ligand.ID']).sort_index()

# Save output to .pickle file
pickle_data = {'ligand_features': encoded_smiles_df,
               'ligand_props': clean_ligands}
pickle_out = open('Ligand_data_clean.pickle', 'wb')
pickle.dump(pickle_data, pickle_out)
pickle_out.close()

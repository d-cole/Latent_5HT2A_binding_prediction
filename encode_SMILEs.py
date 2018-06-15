from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from chemvae.vae_utils import VAEUtils
import pandas as pd
import pickle

# Load VAE trained on zinc SMILEs dataset
vae = VAEUtils(directory='/home/danielcole/Documents/molecules/chemical_vae/models/zinc_properties')
# Load smiles from kiDB 5-HT2A ligand subset
focal_smiles = pd.read_csv('/home/danielcole/Documents/molecules/focal_smiles.csv', index_col=0)
focal_smiles.reset_index(drop=True, inplace=True)

# Encode focal smiles to one-hot encoding
smile_ohots = focal_smiles.apply(axis=1, func=lambda x: vae.smiles_to_hot(x, canonize_smiles=True))
# Run one-hot SMILE encoding through first half of VAE
smile_encode = smile_ohots.apply(func=lambda x: vae.encode(x))
# Decode from latent encoding back to one-hot
smile_decode_ohot = smile_encode.apply(func=lambda x: vae.decode(x))
# Decode from one-hot back to SMILE string
smile_decoded = smile_decode_ohot.apply(func=lambda x: vae.hot_to_smiles(x, strip=True))

# Save all encodings to .pickle file
pickle_data = {'focal_smiles': focal_smiles,
               'smile_onehots': smile_ohots,
               'smile_encode': smile_encode,
               'smile_decode_ohot': smile_decode_ohot,
               'smile_decoded': smile_decoded}

pickle_out = open('/home/danielcole/Documents/molecules/smile_encodings.pickle', 'wb')
pickle.dump(pickle_data, pickle_out)
pickle_out.close()

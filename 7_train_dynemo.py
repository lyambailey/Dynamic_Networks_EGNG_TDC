#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to train DyNeMo on prepared source-localised data
"""
#%% Setup

# Imports
import mne
import numpy as np
import shutil
import os
import glob
import pandas as pd
from scipy.stats import zscore
from osl_dynamics.data import Data
from osl_dynamics.models.dynemo import Config
from osl_dynamics.models.dynemo import Model
from osl_dynamics.models import load
import yaml
import multiprocessing as mp
import pickle
from osl_dynamics.inference import modes
import datetime

mne.set_log_level('ERROR')

# Write out data?
writeOK=True

# Are you running this on WSL?
wsl = True

# Get current date for output files
now = datetime.datetime.now()
datetime_str = now.strftime("%Y-%m-%d_%H%M")

# Read config file and define paths
with open('0_config.yaml', 'r') as file:
    if wsl:
        config_text = file.read().replace('/d/mjt/9/', '/mnt/network/')
        config = yaml.safe_load(config_text)
    else:
        config = yaml.safe_load(file)
    
subjects_dir = config['dirs']['recon_dir']
proc_dir = config['dirs']['proc_dir']
atlas_dir = config['dirs']['atlas_dir']
model_dir = config['dirs']['model_dir']
results_dir = config['dirs']['results_dir']

# Create temporary storage directory for model training
if wsl:
    store_dir = os.path.expanduser("~/osl_tmp")
else:
    store_dir = os.path.join(model_dir, 'tmp')


# Define a helper function to pull the most recent version of a file
# Note that this sleects file based on the date & time that they were last
# modified, not the actual datetime printed in the filename 
def recent_fname(path):
    
    # Takes a partial file path
    files = glob.glob(path)
    most_recent_file = max(files, key=os.path.getmtime)
    
    return most_recent_file

# Load list of subjects
subjects_fname = recent_fname(os.path.join(results_dir, 'subject_order_*.txt'))
subjects = np.loadtxt(subjects_fname, dtype=str).tolist()


#%% Model training

# Loop through subjects, grabbing paths to their prepped data and appending to list
files = []

print(f'Reading data from {len(subjects)} subjects')
for s, subject in enumerate(subjects):
    fname = recent_fname(os.path.join(proc_dir, subject, f'*_{subject}_prepped-raw.fif'))
    files.append(fname)

#%% Do multiple training runs
runs = [1,2,3,4,5] 

for run in runs:
    
    run_lab = f'run{str(run)}'
    print(f'Beginning training run {run}...')
    
    # Ensure a clean temporary storage directory
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir, exist_ok=True)  
    
    # Update model dir
    this_model_dir = f'{model_dir}_{run_lab}'    
    os.makedirs(this_model_dir, exist_ok=True)

    # Load the data
    data = Data(files, sampling_frequency=250, n_jobs=16, load_memmaps=True, 
                store_dir=store_dir)   
    
    # Do time-delay embedding and pca 
    methods = {"filter": {"low_freq": 1, "high_freq": 45},"tde_pca": {"n_embeddings": 15, "n_pca_components": 90},  # 90 components is actually a reduction, because tde multiplied the number of "channels"
               "standardize": {}}
    
    data.prepare(methods)
    print(data)
    
    n_epochs = config['analysis_params']['dynemo_train']['n_epochs']
    n_modes = config['analysis_params']['dynemo_train']['n_modes']
    
    # Define training params
    dynemo_config = Config(
        n_modes=n_modes,
        n_channels=data.n_channels,
        sequence_length=100,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_alpha_temperature=True,
        initial_alpha_temperature=1.0,
        learn_means=False,
        learn_covariances=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=5,
        n_kl_annealing_epochs=10,
        batch_size=16, #16
        learning_rate=0.001,
        n_epochs=20,
    )
    
    model = Model(dynemo_config)
    model.summary()
    
    # Initialisation. 
    init_history = model.random_subset_initialization(data, n_epochs=1, 
                                                      n_init=3, take=0.5, batch_size=8) 
    
    # Full training
    history = model.fit(data)
    
    # Get variational free energy
    free_energy = model.free_energy(data)
    
    # Save model
    if writeOK:
        model.save(os.path.join(this_model_dir, f'{datetime_str}_model'))
        
        # Save free energy 
        history["free_energy"] = free_energy
        pickle.dump(history, open(os.path.join(this_model_dir, "history.pkl"), "wb"))
    
    # Get alpha (time-varying mode mixtures)
    alpha = model.get_alpha(data)
    
    if writeOK:
        os.makedirs(os.path.join(this_model_dir, "inf_params"), exist_ok=True)
        pickle.dump(alpha, open(os.path.join(this_model_dir, "inf_params", "alp.pkl"), "wb"))
    
    # get covs and fix alpha
    means, covs = model.get_means_covariances()
    alpha_rw = modes.reweight_alphas(alpha, covs)
    
    if writeOK:
        np.save(os.path.join(this_model_dir, "inf_params", "covs.npy"), covs)
        pickle.dump(alpha_rw, open(os.path.join(this_model_dir, "inf_params", "alp_rw.pkl"), "wb"))
        
    # Delete the temporary storage directory
    shutil.rmtree(store_dir)
    
    print('###############################################################')
    print(f'{run_lab} Complete!')
    print('###############################################################')

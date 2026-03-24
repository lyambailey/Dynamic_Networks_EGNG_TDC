#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to prepare the preprocessed, annotated and source-localised data for model training with DyNeMo
"""

# Imports
import mne
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import pandas as pd
from scipy.stats import zscore
from osl_dynamics.data import Data
import yaml
import multiprocessing as mp
import datetime

# Write out data?
writeOK=True

# Are you running this on WSL?
wsl = False

# Get current date and time for output files
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

# Define filenames
master_fname = config['filenames']['masterlist_fname'] 

# Other parameters
groups = config['misc']['groups'] 
exclude = config['misc']['exclusion_col']
groups = config['misc']['groups']

# Read masterlist
masterlist = pd.read_excel(master_fname, sheet_name='Sheet1')

# Drop excluded subjects. Do NOT do this if you intend to write back to the masterlist!
masterlist = masterlist[(masterlist[exclude] == 0) & (masterlist['Dx'].isin(groups))]

# Sort by age. 
masterlist = masterlist.sort_values('Age').reset_index()

# Get the list of subjects
subjects = masterlist['SubjectID'].tolist()


# Write out the sorted-by-age list of subjects to preserve subject order. This is essential for post-hoc analyses
if writeOK:
    np.savetxt(os.path.join(results_dir, f'subject_order_{datetime_str}.txt'), subjects, fmt='%s')

# Define a helper function to pull the most recent version of a file
# Note that this sleects file based on the date & time that they were last
# modified, not the actual datetime printed in the filename 
def recent_fname(path):
    
    # Takes a partial file path
    files = glob.glob(path)
    most_recent_file = max(files, key=os.path.getmtime)
    
    return most_recent_file


#%% Prep data

# Define empty list to store data
all_data = []

# Loop through subjects, reading data and appending to list
for subject in subjects:

    print(subject)

    # Files to read. We'll read the most recent version of each file
    raw_fname = recent_fname(os.path.join(proc_dir, subject, f'*_{subject}_source_orth-raw.fif'))

    # Read the raw data 
    raw = mne.io.Raw(raw_fname, preload=True, verbose=False)
    
    # Verify that the raw data has the correct sampling rate
    assert raw.info['sfreq'] == 250, f'ERROR: Sampling rate of raw data is {raw.info['sfreq']} Hz. It should be 250' 
    
    # Crop bad datapoints
    data = raw.get_data(reject_by_annotation='omit', verbose=False)
    
    # Append cropped data to list
    all_data.append(data.T)

# Sign flip everybody's data
data = Data(all_data, 
           sampling_frequency=250, 
           n_jobs = 16) 

# Prepare
methods = {"standardize": {},
           "align_channel_signs": {}}
data.prepare(methods)

# Save the prepared data to disk
data_prepped = data.time_series()
for s, subject in enumerate(subjects):
    
    # Create a new mne.raw object with the prepared data, using the _source_orth-raw info
    raw_fname = recent_fname(os.path.join(proc_dir, subject, f'*_{subject}_source_orth-raw.fif'))
    raw = mne.io.Raw(raw_fname, preload=True, verbose=False)
    raw = mne.io.RawArray(data_prepped[s].T, raw.info)
    
    # Sve the prepped raw to disk. 
    if writeOK:
        raw.save(os.path.join(proc_dir, subject, f'{datetime_str}_{subject}_prepped-raw.fif'), overwrite=True)
      
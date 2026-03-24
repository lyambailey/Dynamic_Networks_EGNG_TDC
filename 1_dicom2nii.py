#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to perform dicom -> nifti conversion, for a handful of subjects
for whom we do not have an existing T1 
"""

# Imports
import os
import glob
import subprocess
import yaml
import multiprocessing as mp
import pandas as pd
import numpy as np

# Read config file
with open('0_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Read masterlist
cols = ['SubjectID', 'Dx', 'T1_dicom_path', config['misc']['exclusion_col']]
cols.append("Has dicom?")  # manual intervention for mopping up remaining subjects without recons. Should ultimately be removed
masterlist = pd.read_excel(config['filenames']['masterlist_fname'], usecols=cols, sheet_name = 'Main_Sheet')

# Drop excluded subjects
masterlist = masterlist[masterlist[config['misc']['exclusion_col']] != 1]

# Only take subjects from our group(s) of interest
# masterlist = masterlist[masterlist['Dx'].isin(config['misc']['groups'])]

# There are still some duplicate SubjectIDs. This really needs to be resolved, but for now
# we'll just drop them. 
masterlist = masterlist.drop_duplicates(subset=["SubjectID"], keep="first")

# Define list of subjects
#subjects = masterlist['SubjectID'].tolist()

# get subjects marked "yes" in the "Has dicom" column
subjects = masterlist.loc[masterlist['Has dicom?']=="yes"]['SubjectID'].tolist()

# Set subjectID as the index for the dataframe
masterlist = masterlist.set_index('SubjectID')

# Loop through subjects, converting dicom to .nii 
for subject in subjects:    
    
    # Define output directory for this subject
    out_dir = os.path.join(config['dirs']['mri_dir'], subject)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # Skip this subject if the output directory already contains files
    if len(os.listdir(out_dir)) >= 1:    
        print(f'Skipping {subject} - T1.nii already exists')
        
    elif len(os.listdir(out_dir)) == 0:
    
        # Get the dicom path
        dicom_path = masterlist.at[subject, 'T1_dicom_path']
        
        if not pd.isna(dicom_path):
        
            print(f'Converting DICOMs from subject {subject}')
        
        # Do the conversion
        command = f"""
            dcm2niix -o {out_dir} {dicom_path}
            
            """
            
        subprocess.run(
            command,
            shell=True, 
            capture_output=True,
            text=True,
            executable='/bin/bash'  # Use bash explicitly
        )
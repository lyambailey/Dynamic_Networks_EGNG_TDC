'''
The purpose of this script is to perform Freesurfer surface reconstruction 
and watershed bem on multiple subjects in parallel 
'''

# Imports
import os
import glob
import subprocess
import yaml
import math
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from mne.bem import make_watershed_bem, make_scalp_surfaces

#%% Define functions

# Define function to search for a T1 .nii file, for a given subject
def find_t1(subject):
    
    # Many T1 mris are stored in Kistina Safar's EGNG folder. Others are stored in 
    # the 'subjects_mri' directory for this project
    mri_path1 = os.path.join('/d','mjt','5','kristinasafar','EGNG','data',subject, 'MRI')
    mri_path2 = os.path.join(config['dirs']['mri_dir'], subject)

    
    # Search both directories for .nii files
    if os.path.exists(mri_path1):
        mri_files = glob.glob(os.path.join(mri_path1,  '*.nii'))
    elif os.path.exists(mri_path2):
        mri_files = glob.glob(os.path.join(mri_path2,  '*.nii'))
    else:
        mri_files = None

        
    # Grab the path to the T1.nii file
    if not mri_files:  # Skip subject if no MRI files found
        return
    elif len(mri_files) > 1:  # There should only be 1 T1 for each subject
        raise ValueError(f"Multiple MRI files found for subject {subject}: {mri_files}")
    else:
        mri_fname = mri_files[0]
        
        return (mri_fname)

# Define function to run recon-all in one subject
def do_recon_all(subject):

    # Make sure this subject does not already have freesurfer output
    if os.path.exists(os.path.join(subjects_dir, subject)):
        #print(f"Subject {subject} already has recon-all output. Skipping.")
        return

    # Grab T1 mri
    mri_fname = find_t1(subject)
    print(mri_fname)
    
    
    if mri_fname is None:
        print(f'No T1 found for {subject}. Skipping...')
        no_t1.append(subject)
        return

    print(f"Processing subject {subject}")


    try:
        # # Source FreeSurfer and run recon-all. Note we can use the following flags to speed up recon-all: -parallel -openmp $N_cores (8 recommended)
        # # Add the following to commands if freesurfer is not already in bashrc path
        # # export FREESURFER_HOME=/usr/local/freesurfer/7.3.2
        # # source $FREESURFER_HOME/SetUpFreeSurfer.sh
        commands = f"""
        recon-all -i {mri_fname} -s {subject} -all -sd {subjects_dir} -parallel -openmp {recon_cores} 
        """
    
        # # recon-all -skullstrip -clean-bm -s {subject} -sd {subjects_dir} # <-- for fixing a bad skull strip
    
        # Run the commands with bash
        subprocess.run(
            commands,
            shell=True, 
            capture_output=True,
            text=True,
            executable='/bin/bash'  # Use bash explicitly
        )
    
        print(f"Recon-all on {subject} complete")
        
    except:
        print(f'Could not perform recon-all for {subject}')
        return

    
# Define function to compute bem surfaces, in one subject
def compute_bem_surfaces(subject):
    
    # Skip if there is no recon folder
    if not os.path.exists(os.path.join(subjects_dir, subject)):
        return
    
    # Ensure this subject does not already have bem surfaces. I think the 'head' 
    # surface is one of the last things that gets created
    if os.path.exists(os.path.join(subjects_dir, subject, 'bem', f'{subject}-head.fif')):
        print('bem surfaces already exist - Skipping')
        return
    
    # Create BEM surfaces
    print(f"Creating BEM surfaces for subject {subject}")

    make_scalp_surfaces(subject=subject, subjects_dir=subjects_dir, overwrite=True)
    make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True, atlas=True, gcaatlas=True)  # using atlases improves accuracy, lowering preflood (default=10) makes the algorithm more aggressive (i.e., likely to exclude more points from the brain)
            
    
    print(f"Finished processing subject {subject}")

#%%
# Process multiple subjects in parallel
if __name__ == "__main__":

    # Read config file
    with open('0_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    # Define the exclusion column in the masterlist
    exclude = config['misc']['exclusion_col']

    # Read list of subjects from the masterlist
    subjects_dir = config['dirs']['recon_dir']
    masterlist = pd.read_excel(config['filenames']['masterlist_fname'], usecols=['SubjectID', 'Dx', exclude, 'T1_path']) 
    masterlist['SubjectID'] = masterlist['SubjectID'].str.strip()

    # Only take subjects from our group(s) of interest
    #masterlist = masterlist[masterlist['Dx'].isin(config['misc']['groups'])]
    masterlist = masterlist[masterlist['Dx'].isin(['ADHD', 'ASD'])]

    # Remove excluded subjects
    masterlist = masterlist[masterlist[exclude] == 0]
    
    # Get the list of (included) subjects 
    subjects = masterlist['SubjectID'].tolist() 

    print(f'processing {len(subjects)} subjects')

    # Define an empty list to keep track of subjects without a T1
    no_t1 = []
   
    # Allocate N cores for each recon 
    recon_cores = config['recon-params']['n_cores_per_recon']

    # Allocate cores for parallel processing
    parallel_cores = int(np.round(mp.cpu_count()/2))  #recon_cores))  # this spreads cores sensibly across subjects, considering N cores will be allocated for each
    print(f'Subjects parallelized across {parallel_cores} cores. {recon_cores} cores will be allocated per recon.')

    # Create workers
    pool = mp.Pool(processes=parallel_cores)

    # Do the work...
    
    # Recon-all
    for _ in tqdm(pool.imap_unordered(do_recon_all, subjects), total=len(subjects)):
        pass
    
    # # Watershed bem
    # for _ in tqdm(pool.imap_unordered(compute_bem_surfaces, subjects), total=len(subjects)):
    #     pass

    # Alternatively we can pass individual subjects
    # do_recon_all('CC04')
    # compute_bem_surfaces('CC04')



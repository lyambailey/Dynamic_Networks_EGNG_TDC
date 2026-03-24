'''
The purpose of this script is to perform coregistration on a user-defined subject. 
The script is split into three main stages:
    1. Setup (load packages, define file paths, etc)
    2. Read MRI fiduical points from the masterlist and convert to MNE_friendly fiducials.fif format
    3. Perform coregistraion using fiducial points and refine with ICP

'''

#%% Setup 

# Imports
import os
import pandas as pd
import mne
import yaml
import numpy as np
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import matplotlib.pyplot as plt
from mne.io.constants import FIFF
from mne.transforms import _coord_frame_name
import math

mne.viz.set_3d_backend("pyvistaqt")

writeOK=True

# Read config file and define paths
with open('0_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
# Define paths
data_path = config['dirs']['raw_data_dir']
subjects_dir = config['dirs']['recon_dir']
proc_dir = config['dirs']['proc_dir']

# Define filnames
master_fname = config['filenames']['masterlist_fname']  # contains demographics, raw data path and fiducial coordinates for each subject

# Define the exclusion and groups (of interest) column in the masterlist
exclude = config['misc']['exclusion_col']
groups = config['misc']['groups']

# Read masterlist
cols = ['SubjectID',                    # unique ID for each scan
        exclude,                        # ==1 if excluded
        'Age',                          # age in years
        'Dx',                           # Clinical diagnosis
        'VigilancePath',                # path to raw ctf file (EGNG task)
        'surrogate_mri',                # ==1 if we're using a different subject's mri for coreg
        'surrogate_mri_id',             # The subject whose mri we're using as a surrogate, when applicable
        'fiducial_lpa_tkreg-RAS',       # mri coordinate (mm) for the lpa
        'fiducial_rpa_tkreg-RAS',       # mri coordinate (mm) for the rpa
        'fiducial_nas_tkreg-RAS',       # mri coordinate (mm) for the nasion
        'fiducial_weights_nas_lpa_rpa'  # fiducial weights for head-mri alignment
        ]

masterlist = pd.read_excel(master_fname, usecols=cols)

# Get the list of (included) subjects 
subjects = masterlist[(masterlist[exclude] == 0) & (masterlist['Dx'].isin(groups))]['SubjectID'].tolist()

# Set subjectID as the index
masterlist.set_index('SubjectID', inplace=True) 

#%% Define coregistration function
def do_coreg(subject, custom_weights=None):  # allows modification of fiducial weights if needed
   
    
    # Define subject-specific directories and filenames
    out_dir = os.path.join(proc_dir, subject)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Subject-specific files to read 
    raw_ctf_fname = masterlist.loc[subject, 'VigilancePath'] 
    fids_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-fiducials.fif')
    
    # Files to write
    raw_fif_fname = os.path.join(out_dir, os.path.splitext(os.path.basename(raw_ctf_fname))[0] + '-raw.fif')
    fid_fif_fname = os.path.join(subjects_dir, subject, 'bem', f'{subject}-mri_fiducials.fif')
    trans_fif_fname = os.path.join(out_dir, subject + '-trans.fif')
    
    # Load raw data
    raw = mne.io.read_raw_ctf(raw_ctf_fname, preload=True)
    info = raw.info
    
    # Define subject_mri. This is usually the same as subject, but in some cases
    # (e.g. where that subject's mri was poor quality or missing) we'll use a different 
    # subject's mri as a "surrogate" for coregistration purposes
    surrogate = masterlist.loc[subject, 'surrogate_mri']
    if surrogate == 0:
        subject_mri = subject
    elif math.isnan(surrogate):
        subject_mri = subject
    elif surrogate == 1:
        subject_mri = masterlist.loc[subject, 'surrogate_mri_id']
        
        # Update fid_fif_fname to point to the surrogate's recon folder
        fid_fif_fname = os.path.join(subjects_dir, subject_mri, 'bem', f'{subject_mri}-mri_fiducials.fif')
        
        # Update trans_fif_fname such that transformations will be output to the subject's proc data folder, 
        # but be labelled with subect_mri
        trans_fif_fname = os.path.join(out_dir, subject_mri + '-trans.fif')
        
    # Check that subject_mri has a recon folder and bem dir
    assert os.path.exists(os.path.join(subjects_dir, subject_mri)), (
        f"No recon folder found for subject {subject_mri}. Cannot proceed with coregistration."
        )
    
    assert os.path.exists(os.path.join(subjects_dir, subject_mri, 'bem')), (
        f"No BEM surfaces found for subject {subject_mri}. Cannot proceed with coregistration." 
        )

    # Read subject_mri's fiducials from masterlist
    nas = [float(x.strip()) for x in masterlist.loc[subject_mri, 'fiducial_nas_tkreg-RAS'].split(',')]
    lpa = [float(x.strip()) for x in masterlist.loc[subject_mri, 'fiducial_lpa_tkreg-RAS'].split(',')]
    rpa = [float(x.strip()) for x in masterlist.loc[subject_mri, 'fiducial_rpa_tkreg-RAS'].split(',')]
    
    fids = {"nasion": nas, "lpa": lpa, "rpa": rpa}
    
    # Check that the x coordinate of the lpa is negative
    assert fids['lpa'][0] < 0, "The x coordinate for the lpa is non-negative. Are you sure these coords are correct?"
    

    # Construct fiducial list in MNE format
    fiducials_list = [
        dict(kind=FIFF.FIFFV_POINT_CARDINAL,
             ident=FIFF.FIFFV_POINT_NASION,
             r=np.array(fids['nasion'])/ 1000,  # mm → m
             coord_frame=FIFF.FIFFV_COORD_MRI),
    
        dict(kind=FIFF.FIFFV_POINT_CARDINAL,
             ident=FIFF.FIFFV_POINT_LPA,
             r=np.array(fids['lpa'])/ 1000,  # mm → m
             coord_frame=FIFF.FIFFV_COORD_MRI),
    
        dict(kind=FIFF.FIFFV_POINT_CARDINAL,
             ident=FIFF.FIFFV_POINT_RPA,
             r=np.array(fids['rpa'])/ 1000,  # mm → m
             coord_frame=FIFF.FIFFV_COORD_MRI),
    ]
    
    # Write fiducials to disk
    mne.io.write_fiducials(fid_fif_fname, fiducials_list, coord_frame='mri', overwrite=True)
    
    # Load the fids back in and check their positions
    fids_mri, coord_frame = mne.io.read_fiducials(fid_fif_fname)
    
    print(f"Coordinate frame: {_coord_frame_name(coord_frame)}")
    for f in fids_mri:
        print(f)
        
    
    # Plot alignment with fiducials
    mne.viz.plot_alignment(
        subject=subject_mri,
        subjects_dir=subjects_dir,
        surfaces=['head-dense'],  # show only head surface
        coord_frame='mri',
        meg=False,
        eeg=False,
        mri_fiducials=fids_mri  
    )
    
    # Perform coregistration
    coreg = mne.coreg.Coregistration(info=info, subject=subject_mri, fiducials=fids_mri,
                                     subjects_dir=subjects_dir) 
    
    # Fit fiducials. Weights are defined in the masterlist. Default is equal weights
    # [1.0, 1.0, 1.0], but these can be adjusted if, for example, one fiducial coil 
    # is missing
    if custom_weights is not None:
        weights=custom_weights
    else:
        weights = [float(x.strip()) for x in masterlist.loc[subject_mri, 'fiducial_weights_nas_lpa_rpa'].split(',')]
               
    coreg = coreg.fit_fiducials(nasion_weight=weights[0],
                                lpa_weight=weights[1],
                                rpa_weight=weights[2])  
    
    # Refine coreg with icp
    coreg = coreg.fit_icp()
    
    # Plot the results
    plot_kwargs = dict(
        subject=subject_mri,
        subjects_dir=subjects_dir,
        surfaces=("head-dense"),
        dig=True,
        meg=("helmet"),
        mri_fiducials=fids_mri,
        show_axes=True
    )
     
    mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)  
    
    # Write raw data and mri <-> head transformation matrix, both in .fif format
    if writeOK:
        raw.save(raw_fif_fname, overwrite=True)
        mne.write_trans(trans_fif_fname, coreg.trans, overwrite=True)
    
    return [fids, raw, weights, subject_mri, coreg.trans]



#%% Perform coregistration on each subject
subject_n = subjects[0]
fids, raw, weights, subject_mri, trans = do_coreg(subjects[subject_n], custom_weights = None)



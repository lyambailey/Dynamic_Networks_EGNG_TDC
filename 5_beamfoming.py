#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to perform source localization on preprocessed and annotated MEG data

"""

# Imports
import mne
from mne_connectivity import symmetric_orth
import os
import glob
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import zscore, iqr
from nilearn import plotting, datasets, image, surface
import nibabel as nib
from scipy.ndimage import uniform_filter1d, label
import pandas as pd
import yaml
import csv
import pandas as pd
import re
import multiprocessing as mp
import datetime
from collections import Counter

mne.set_log_level('ERROR')

# Write output?
writeOK = True

# Get current date for output files
now = datetime.datetime.now()
datetime_str = now.strftime("%Y-%m-%d_%H%M")

# Read config file and define paths
with open('0_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
subjects_dir = config['dirs']['recon_dir']
proc_dir = config['dirs']['proc_dir']
atlas_dir = config['dirs']['atlas_dir']

# Define filenames
master_fname = config['filenames']['masterlist_fname'] 
atlas_fname = os.path.join(atlas_dir, 'MEG_atlas_38_regions_4D.nii.gz')
atlas_labels_fname = os.path.join(atlas_dir, 'MEG_atlas_38_regions_names.csv')
coords_fname = os.path.join(atlas_dir, 'MEG_atlas_38_regions_coords.npy')

# Read 38 region atlas
atlas = image.load_img(atlas_fname)
coords = np.load(coords_fname)
names = np.squeeze(pd.read_csv(atlas_labels_fname, header=None).to_numpy())

# Other parameters
groups = config['misc']['groups'] 
filt_params = config['preproc_params']['filtering']
resampling_freq = config['preproc_params']['resampling_freq']
exclude = config['misc']['exclusion_col']
groups = config['misc']['groups']

# Read masterlist
masterlist = pd.read_excel(master_fname)

# Get the list of (included) subjects
subjects = masterlist[(masterlist[exclude] == 0) & (masterlist['Dx'].isin(groups))]['SubjectID'].tolist()

# Set subject ID as the index
masterlist = masterlist.set_index('SubjectID')

#%% Define functions
def make_4d_atlas_nifti(atlas_img, values):

    # load fsaverage and atlas   
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()

    # place values in each parcel region
    regs = []
    for reg in range(atlas_data.shape[-1]):
        atlas_reg = atlas_data[:,:,:,reg]
        atlas_reg[atlas_reg>0] = 1
        regs.append(atlas_reg * values[reg])
    atlas_new = np.sum(regs, 0)

    # make image from new atlas data
    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)
    
    return img_interp

def surface_brain_plot(img, subjects_dir, surf='inflated', cmap='RdBu_r', symmetric=True, 
                       threshold=0, fade=True, cbar_label=None, figsize=(10,7)):

    
    # make MNE stc out of nifti
    lh_surf = os.path.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
    lh = surface.vol_to_surf(img, lh_surf)
    rh_surf = os.path.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
    rh = surface.vol_to_surf(img, rh_surf)
    data = np.hstack([lh, rh])
    vertices = [np.arange(len(lh)), np.arange(len(rh))]
    stc = mne.SourceEstimate(data, vertices, tmin=0, tstep=1)

    # set up axes
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_axes([0, 0.60, 0.35, 0.35])  # top-left
    ax2 = fig.add_axes([0.65, 0.60, 0.35, 0.35])  # top-right
    ax3 = fig.add_axes([0.0, 0.15, 0.35, 0.35])  # bottom-left
    ax4 = fig.add_axes([0.65, 0.15, 0.35, 0.35])  # bottom-right
    ax5 = fig.add_axes([0.32, 0.3, 0.36, 0.5])  # center 
    cax = fig.add_axes([0.25, 0.1, 0.5, 0.03]) # colorbar ax
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/2)
        if fade:
            clim = {'kind': 'value', 'pos_lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'pos_lims':(threshold, threshold, vmax)}
    else:
        vmax = np.max(data)
        vmin = np.min(data)
        mid = threshold + ((vmax-threshold)/3)
        if fade:
            clim = {'kind': 'value', 'lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'lims':(threshold, threshold, vmax)}
        
    if surf=='inflated':
        cortex='low_contrast'
    elif surf=='pial':
        cortex=(0.6, 0.6, 0.6)
    else:
        cortex=(0.6, 0.6, 0.6)
    plot_kwargs = dict(subject='fsaverage',
                       subjects_dir=subjects_dir,
                       surface=surf,
                       cortex=cortex,
                       background='white',
                       colorbar=False,
                       time_label=None,
                       time_viewer=False,
                       transparent=True,
                       clim=clim,
                       colormap=cmap,
                       )
    
    def remove_white_space(imdata):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped

    # top left
    views = ['lat']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax1.imshow(screenshot)

    # top right
    views = ['lat']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax2.imshow(screenshot)

    # bottom left
    views = ['med']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax3.imshow(screenshot)

    # bottom right
    views = ['med']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax4.imshow(screenshot)

    # middle
    views = ['dorsal']
    hemi = 'both'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    background = np.sum(screenshot, -1) == 3*255
    alpha = np.ones(screenshot.shape[:2])  
    alpha[background] = 0
    ax5.imshow(screenshot, alpha=alpha)

    # colorbar
    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)

    
    return fig

def finish_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(alpha=0.3)
    
def isoutlier(data, thresh=5):
    data = np.asarray(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr_value = iqr(data)
    lower_bound = q25 - thresh * iqr_value
    upper_bound = q75 + thresh * iqr_value
    outliers = data > upper_bound
    return outliers


#%% Define a convenience function for beamforming on a single subject. This is a helpful
# alternative to running all subjects in a loop directly, in the event that we need to re-run specific subjects
def do_beamforming(subject):

    print(subject)
  
    # Use a try tatement here to avoid single-subject errors breaking the whole workflow
    # when the function is passed to a loop
    try:
        # Define subject_mri. This is usually subject, except where we have used a surrogate
        # for coregistration        
        surrogate = masterlist.loc[subject, 'surrogate_mri_id']
        
        if pd.isna(surrogate):
            subject_mri = subject
        else:
            subject_mri = surrogate

        # Files to read
        raw_fname = glob.glob(os.path.join(proc_dir, subject, '*-raw-filt-annotated.fif'))[0]  # preprocessed raw data
        trans_fname = os.path.join(proc_dir, subject, subject_mri + '-trans.fif') 
        
        # Files to write 
        source_raw_fname = os.path.join(proc_dir, subject, f'{datetime_str}_{subject}_source-raw.fif')
        source_orth_raw_fname = os.path.join(proc_dir, subject, f'{datetime_str}_{subject}_source_orth-raw.fif')

        # Output directory for figures (for QA)
        subj_figs_dir = os.path.join(proc_dir, subject, 'figs') # figures
        
        # Read the preprocessed raw data and head<->MRI transform. 
        raw = mne.io.read_raw_fif(raw_fname, preload=True) 
        trans = mne.read_trans(trans_fname)

        ## Setup for beamformer...
        
        # Define single-shell conduction model
        conductivity = (0.3,) 
        
        # Create bem model
        bem = mne.make_bem_model(subject_mri, subjects_dir=subjects_dir, conductivity=conductivity)
        bemSol = mne.make_bem_solution(surfs=bem)
        
        # Get mri-->MNI transform and apply inverse to atlas
        mri_mni_t = mne.read_talxfm(subject_mri, subjects_dir=subjects_dir)['trans']
        mni_mri_t = np.linalg.inv(mri_mni_t)
        centroids_mri = mne.transforms.apply_trans(mni_mri_t, coords / 1000) # in m
        
        # Create atlas source space
        rr = centroids_mri # positions
        nn = np.zeros((rr.shape[0], 3)) # normals
        nn[:,-1] = 1.
        src = mne.setup_volume_source_space(
            subject_mri,
            pos={'rr': rr, 'nn': nn},
            subjects_dir=subjects_dir,
            verbose=True,
            n_jobs = -1
        )
        
        # Make forward solution 
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bemSol, meg=True)
                                                                            
        # Compute raw data covariance
        cov = mne.compute_raw_covariance(raw, reject_by_annotation=True)
        cov_data = cov.data				
        
        # Set up for beamformer
        filters = mne.beamformer.make_lcmv(
                raw.info,
                fwd,
                cov,
                reg=0.05,
                noise_cov=None,
                pick_ori='max-power',
                rank=None,
                reduce_rank=True,
                verbose=False,
                )
        
        # Apply the beamformer and extract data
        stc =  mne.beamformer.apply_lcmv_raw(raw, filters, verbose=False)
        source_data = stc.data
        
        # Create an mne.Raw container for the beamformed data
        info = mne.create_info(list(names), raw.info['sfreq'], 'misc', verbose=False)
        source_raw = mne.io.RawArray(source_data, info, verbose=False)
        source_raw.set_meas_date(raw.info['meas_date'])
        source_raw.set_annotations(raw.annotations, verbose=False)
        
        # Do symmetruc orthogonalisation to reduce source leakage
        source_data_orth = zscore(symmetric_orth(source_data), 1)
        source_raw_orth = mne.io.RawArray(source_data_orth, source_raw.info)
        source_raw_orth.set_meas_date(raw.info['meas_date'])
        source_raw_orth.set_annotations(raw.annotations, verbose=False)
        
        # Save results 
        if writeOK:

            # If source_raw_fname exists, append a number to make it unique
            if os.path.exists(source_raw_fname):
                base, ext = os.path.splitext(source_raw_fname)
                i = 1
                while os.path.exists(f"{base}_{i}{ext}"):
                    i += 1
                source_raw_fname = f"{base}_{i}{ext}"
            source_raw.save(source_raw_fname, overwrite=True)
            source_raw_orth.save(source_orth_raw_fname, overwrite=True)

        # Plot output for QA
        if writeOK:

            # Covariance matrix
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(cov_data)
            ax.set_xlabel('Sensor')
            ax.set_ylabel('Sensor')
            ax.set_title('Data Covariance')
            plt.tight_layout()
            fig.savefig(os.path.join(subj_figs_dir, subject + f'_data_cov_{datetime_str}.png'))
            plt.close()


            # Source level PSD
            psd = source_raw_orth.compute_psd(method='welch', fmin=1, fmax=100, n_fft=int(info['sfreq']), # 1 s window
                                            picks='all')
            freqs = psd.freqs
            psd_data = psd.data
            
            # plot PSD
            psd_data = 10 * np.log10(psd_data)
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(freqs, psd_data.T, color='gray', alpha=0.2)
            ax.plot(freqs, np.mean(psd_data,0), color='black', linewidth=2.5)
            ax.set_xlabel('Frequency (Hz)', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.set_title('Source-Level PSD', fontsize=14)
            finish_plot(ax)
            plt.tight_layout()
            fig.savefig(os.path.join(subj_figs_dir, subject + f'_source_psd_{datetime_str}.png'))
            plt.close()


            # Atlas source space

            # Optionally specify slice indices to get a closer look for individual subjects.
            # May need adjustment. 
            slices_sag=list(range(0, 200, 2)) 
            slices_cor=list(range(0, 200, 2))
            slices_axi=list(range(50, 250, 2))

            # Note that a high number of slices can cause rendering problems. 
            # If you see an incomplete image, simply re-run the code
            cor = mne.viz.plot_bem(subject=subject_mri, subjects_dir=subjects_dir,
                            src=src, orientation='coronal', show=False, slices=slices_cor) 
            
            sag = mne.viz.plot_bem(subject=subject_mri, subjects_dir=subjects_dir, 
                            src=src, orientation='sagittal', show=False, slices=slices_sag)
            
            axi = mne.viz.plot_bem(subject=subject_mri, subjects_dir=subjects_dir, 
                            src=src, orientation='axial', show=False, slices=slices_axi) 
            
            # Save the output 
            cor.savefig(os.path.join(subj_figs_dir, subject + f'_src_coronal_{datetime_str}.png'))
            plt.close()
            sag.savefig(os.path.join(subj_figs_dir, subject + f'_src_sagittal_{datetime_str}.png'))
            plt.close()
            axi.savefig(os.path.join(subj_figs_dir, subject + f'_src_axial_{datetime_str}.png'))
            plt.close()
    except:
        print(f"Beamforming failed for subject {subject}")


#%% Loop through subjects, running beamformer function for each
for subject in subjects:
    do_beamforming(subject)

#%% Optionally run the function on one subject at a time
subject_n = 0
do_beamforming(subjects[subject_n])






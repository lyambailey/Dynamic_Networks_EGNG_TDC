#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to perform basic preprocessing on MEG data from the emotional go-nogo (EGNG) "Vigilance"task.
Preprocessing output (e.g., amount of head movement, number of bad timepoints, etc) for each subject is saved to the masterlist
"""

# Imports
import mne
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import iqr
from scipy.ndimage import label
import pandas as pd
import yaml
import pandas as pd
import datetime
import math

mne.set_log_level('ERROR') # suppress MNE terminal output

# Write output?
writeOK = True

# Read config file and define paths
with open('0_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
subjects_dir = config['dirs']['recon_dir']
proc_dir = config['dirs']['proc_dir']
atlas_dir = config['dirs']['atlas_dir']

# Define filenames
master_fname = config['filenames']['masterlist_fname'] 

# Other parameters
groups = config['misc']['groups'] 
filt_params = config['preproc_params']['filtering']
resampling_freq = config['preproc_params']['resampling_freq']
exclude = config['misc']['exclusion_col']
groups = config['misc']['groups']

# Read masterlist and define subjects
masterlist_orig = pd.read_excel(master_fname)
masterlist = masterlist_orig.copy()

subjects = masterlist[(masterlist[exclude] == 0) & (masterlist['Dx'].isin(groups))]['SubjectID'].tolist()

# Set subjectID as the masterlist index (this allows us to write preprocessing output to specific rows later on)
masterlist = masterlist.set_index('SubjectID')

# Make a backup of the original masterlist, since we're going to overwrite it at the end of this script
if writeOK:
    now=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    masterlist.to_excel(
        f'/d/mjt/9/projects/Dynamo-Analysis/POND-EGNG/support_files/master_backups/masterlist_backup_{now}.xlsx',
        index=False)

# Define functions
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


# Loop through subjects, loading in raw data and doing preprocessing
for i, subject in enumerate(subjects):
    print(subject)

    # Files to read
    raw_ctf_fname = masterlist[masterlist[exclude]==0].loc[subject, 'VigilancePath']
    
    # Files to write 
    raw_preproc_fname = os.path.join(proc_dir, subject, f'{subject}-EGNG25-raw-filt-annotated.fif')

    # Define output directory for subject-specific figures (used for QA)
    subj_figs_dir = os.path.join(proc_dir, subject, 'figs') 
    if not os.path.exists(subj_figs_dir):
        os.mkdir(subj_figs_dir)
    
    # Read the raw CTF data 
    try:
        raw = mne.io.read_raw_ctf(raw_ctf_fname, preload=True)
    except:
        print(f'WARNING: could not read raw data for {subject}, skipping subject')
        continue

    # Get head position coil data
    chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)
    head_xyz = head_pos[:,1:4]*100 - head_pos[0,1:4]*100  
    times = head_pos[:,0]

    # Apply band-pass filtering
    raw.filter(l_freq=filt_params['lfreq'], h_freq=filt_params['hfreq'])
    
    # Pick mags only and resample
    raw.pick('mag')
    raw = raw.resample(resampling_freq)
    
    # annotate segments with high movement
    threshold = 10 # mm
    bad_mask = np.any(np.abs(head_xyz) > threshold, 1) 
    bad_labels, _ = label(bad_mask)
    bad_times = []
    for cluster in np.unique(bad_labels[bad_labels>0]):
        cluster_times = times[bad_labels==cluster]
        time0, time1 = cluster_times[0], cluster_times[-1]
        bad_times.append([time0, time1])
      
    # join together bad segments within threshold
    if len(bad_times) > 1:
        keep_threshold = 20 # seconds. Joins together bad segments within this time
        bad_times_joined = []
        bad_times_joined.append(bad_times[0])
        for i in np.arange(1, len(bad_times)):
            gap = bad_times[i][0] - bad_times[i-1][1]
            if gap < keep_threshold:
                bad_times_joined[-1][1] = bad_times[i][1].copy()
            else:
                bad_times_joined.append(bad_times[i])
        bad_times = bad_times_joined.copy()
        del bad_times_joined
        
    # annotate
    annot = raw.annotations
    for bad_time in bad_times:
        onset = bad_time[0]
        duration = bad_time[1] - bad_time[0]
        description = 'BAD_pos'
        annot += mne.Annotations(onset, duration, description, orig_time=annot.orig_time)
    raw.set_annotations(annot, verbose=False)
    
    # remove bad channels
    chan_var = np.var(raw.get_data(), axis=1)
    outliers = isoutlier(chan_var)
    bad_chan = [raw.ch_names[ch] for ch in range(len(raw.ch_names)) if outliers[ch]]
    raw.info['bads'] = bad_chan
    fig = raw.plot_sensors(show=False)
    fig.savefig(os.path.join(subj_figs_dir, subject + '_bad_sensors.png'))
    plt.close()
    raw.drop_channels(bad_chan)
    
    
    # annotate bad segments (treat as resting state)
    data = raw.get_data()
    segment_len = int(1 * raw.info['sfreq'])
    variances = np.array([np.mean(np.var(data[:, i:i+segment_len], axis=1)) for i in np.arange(0, data.shape[1]-segment_len+1, segment_len)])
    outliers = isoutlier(variances, thresh=3)
    annot = raw.annotations
    for i, ind in enumerate(np.arange(0, data.shape[1]-segment_len, segment_len)):
        if outliers[i]:
            onset = raw.times[ind]
            duration = segment_len * (1/raw.info['sfreq'])
            description = 'BAD_var'
            annot += mne.Annotations(onset, duration, description, orig_time=annot.orig_time)
    raw.set_annotations(annot, verbose=False)
    
    # Get preprocessing stats
    n_orig = len(raw.times) # N timepoints in original raw 
    n_bad_chans = len(bad_chan)  # N bad channels
    n_bad_var = np.sum(annot.description=='BAD_var') # N timepoints exceeding variance threshold
    n_bad_pos = np.sum(annot.description=='BAD_pos') # N timepoints exceeding movement threshold
    mean_mov = np.mean(np.abs(head_xyz)) # mean movement in any direction
    max_mov = np.max(np.abs(head_xyz)) # Maximum movement in any direction 
    percent_dropped = (((n_bad_var + n_bad_pos) * raw.info['sfreq']) / n_orig) * 100  # total percentage of dropped timepoints  
    
    # Save preprocessed raw to disk and store preprocessing stats
    if writeOK:
        raw.save(raw_preproc_fname, overwrite=True)

    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'N_bad_chans'] = n_bad_chans
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'N_bad_timepoints_var'] = n_bad_var
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'Movement_thresh'] = threshold
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'Mean_movement'] = mean_mov
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'Max_movement'] = max_mov
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'N_bad_timepoints_pos'] = n_bad_pos
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'N_orig_timepoints'] = n_orig
    masterlist.loc[(masterlist[exclude] == 0) & (masterlist.index == subject), 'percent_lost_timepoints'] = percent_dropped

    # Plot preprocessing output
    if writeOK:

        # Head position
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(times, head_xyz[:,0], label='x')
        ax.plot(times, head_xyz[:,1], label='y')
        ax.plot(times, head_xyz[:,2], label='z')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Position (mm)', fontsize=12)
        ax.set_title('Head Position', fontsize=14)
        ax.legend()
        finish_plot(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(subj_figs_dir, subject + '_head_movement.png'))
        plt.close()
        
        # Sensor-level PSD 
        psd = raw.compute_psd(fmin=0, fmax=45,
                               reject_by_annotation=True)
        psd_data, freqs = psd.data, psd.freqs
        psd_data = 10 * np.log10(psd_data)
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(freqs, psd_data.T, color='gray', alpha=0.2)
        ax.plot(freqs, np.mean(psd_data,0), color='black', linewidth=2.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Sensor-Level PSD', fontsize=14)
        fig.savefig(os.path.join(subj_figs_dir, subject + '_sensor_psd.png'))
        plt.close()
        
        # Sensor power
        fig, ax = plt.subplots(1,3, figsize=(10,4))
        bands = {'Theta (3-7 Hz': (3, 7), 'Alpha (8-12 Hz)': (8, 12), 'Beta (13-30 Hz)': (13, 30)}
        psd.plot_topomap(bands, axes=ax, normalize=True)
        plt.tight_layout()
        fig.savefig(os.path.join(subj_figs_dir, subject + '_sensor_power.png'))
        plt.close()
    


# Write out the masterlist containing preproc output. Note that individual
# rows are updated on each loop of subjects, so we do not need to run all 
# subjects at once!
preproc_output = masterlist.copy().reset_index()

if writeOK:
    preproc_output.to_excel(master_fname, index=False)


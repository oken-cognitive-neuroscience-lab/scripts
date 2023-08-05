"""Gamma analysis for data collected from the cortiQ device."""
import os

import logging
import argparse
import numpy as np

from typing import List, Tuple, Dict

from bcipy.helpers.load import load_experimental_data
from scipy.signal import hilbert

from BCI2kReader import BCI2kReader as bci2k

from bcipy.signal.process.decomposition.psd import (
    power_spectral_density, PSD_TYPE)
# from bcipy.signal.process.decomposition import continuous_wavelet_transform
from bcipy.signal.process.filter import Notch

import mne
from mne.io import read_raw_bdf
from mne.viz import plot_compare_evokeds
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
CONDITIONS = ['Letter1', 'Letter2', 'Letter3', 'Letter4', 'DigitSpanWM1', 'DigitSpanWM2', 'DigitSpanWM3', 'DigitSpanWM4']


def load_data_dat(session: str) -> Tuple[np.ndarray, List[float], int, List[str]]:
    """Load data from a dat file.

    Parameters:
        session - path to session directory

    Returns:
        data - data from the session (channels X samples)
        labels - labels for the session
        fs - sampling frequency
    """
    # find a single .dat file in the session directory
    dat_files = [f for f in os.listdir(session) if f.endswith('.dat')]
    if len(dat_files) != 1:
        raise Exception(f'Expected a single .dat file in {session}, found {len(dat_files)}')
    dat_file = f"{session}\{dat_files[0]}"

    # load data
    raw = bci2k.BCI2kReader(dat_file)

    # extract the start time for each trial from the raw.states["StimulusBegin"]. 
    # The first instance of the stimulus is the start of the trial and is 1.0 for a few samples, 
    # then back to 0.0 until the next trail starts (1.0)
    labels = []
    duration = False
    for i, s in enumerate(raw.states["StimulusBegin"][0]):
        if s == 1.0 and not duration:
            labels.append(i)
            duration = True
        elif s == 0.0 and duration:
            duration = False

    # print number of channels
    print(f'Number of channels: {raw.signals.shape[0]}')
    # print recording length in seconds
    print(f'Recording length: {raw.signals.shape[1] / raw.samplingrate} seconds')
        
    return raw.signals, labels, raw.samplingrate, raw.parameters['ChannelNames']


def load_data_bdf(session: str) -> Tuple[np.ndarray, List[float], int]:
    """Load data from a bdf file.

    Parameters:
        session - path to session directory

    Returns:
        data - data from the session (channels X samples)
        labels - labels for the session
        fs - sampling frequency
    """
    # find a single .bdf file in the session directory
    bdf_files = [f for f in os.listdir(session) if f.endswith('.bdf')]
    if len(bdf_files) != 1:
        raise Exception(f'Expected a single .bdf file in {session}, found {len(bdf_files)}')
    bdf_file = f"{session}\{bdf_files[0]}"

    # load data
    raw = read_raw_bdf(bdf_file, preload=True)
    data = raw.get_data()
    fs = raw.info['sfreq']
    # load labels from data annotations
    labels = []
    for a in raw.annotations:
        labels.append((a['onset'], a['duration'], a['description']))
    return data, labels, fs


def reshape_data_into_trials(data: np.ndarray, labels: List[float], post_stim: float, pre_stim: float, interval: int, fs: int) -> np.ndarray:
    """Reshape data into trials. 
    
    Returns:
        data Channels X Trials X Samples or Channels X Trials X Intervals X Samples (if interval < post_stim)"""
    # turn into samples
    pre_stim = int(pre_stim * fs)
    post_stim = int(post_stim * fs)
    interval = int(interval * fs)

    # calculate the number of intervals in the window and window length
    window_length = pre_stim + post_stim
    
    assert window_length % interval == 0, f'Window length {window_length} is not divisible by interval {interval}.'
    intervals_in_window = int(window_length / interval)

    trials = []

    if intervals_in_window == 1:
        for label in labels:
            start = label - pre_stim
            stop = label + post_stim
            trials.append(data[:, start:stop])
        trials = np.array(trials)
        return trials

    # correct labels to start and stop give the prestimulus and poststimulus time
    for label in labels:
        # label == 2000
        start = label - pre_stim # 50 == 2000 - 50 == 1950
        stop = label + post_stim # 50 == 2000 + 50 == 2050
        intervals = [] 
        tmp_interval = 0
        for i in range(intervals_in_window): # 4
            start_offset = start + tmp_interval # 1950 + 0 == 1950
            intervals.append(data[:, start_offset: start_offset + interval])  # 1950:1950+50 == 1950:2000
            tmp_interval += interval # 0 + 50 == 50
        trials.append(intervals) 

    # reshape into channels X trials X X intervals X samples
    trials = np.array(trials) # Trials X Intervals X Channels X Samples
    trials = np.transpose(trials, (2, 0, 1, 3)) # Channels X Trials X Intervals X Samples
    return trials


def calculate_fft(data, fs, trial_length, freq_range=(50, 80), relative=False):
    """Calculate FFT Gamma
    Calculate the amount of gamma using FFT.
    """
    return power_spectral_density(
                data,
                freq_range,
                sampling_rate=fs,
                window_length=trial_length,
                method=PSD_TYPE.WELCH,
                plot=False,
                relative=relative)


def calculate_gamma(data: np.ndarray, fs: int, trial_length: int, freq_range=(50, 80), relative=False) -> np.ndarray:
    """Calculate gamma power for each trial.

    Parameters:
        data - data from the session (channels X trials X intervals X samples)
        fs - sampling frequency
        trial_length - length of each trial in samples
        freq_range - range of frequencies to calculate gamma for
        relative - calculate relative gamma

    Returns:
        gamma - gamma power for each trial (channels X trials X intervals)
    """
    # calculate gamma for each channel
    gamma = []
    for channel in data: # data is in the shape of channels X trials X intervals X samples
        sub_gamma = []
        for trial in channel: # trial is in the shape of trials X intervals X samples
            sub_sub_gamma = []
            for interval in trial:
                sub_sub_gamma.append(calculate_fft(interval, fs, trial_length, freq_range, relative))
            sub_gamma.append(sub_sub_gamma)
        gamma.append(sub_gamma)

    return np.array(gamma)


def preprocess_data(data: np.ndarray, labels: List[float], fs:int, notch_filter: int = 60):
    """Preprocess data.

    Parameters:
        data - data from the session (channels X trials x samples)
        labels - labels from the session (onset, duration, description)
        fs - sampling frequency

    Returns:
        data - preprocessed data (channels X trials X samples)
    """
    # get mean voltage for all channels and trials
    mean_voltage = np.mean(data, axis=2)

    # correct for mean voltage across channels and trials
    for channel in data:
        for trial in channel:
            trial -= mean_voltage
            # filter data with a notch filter
            trial, _ = Notch(fs, notch_filter)(trial, fs)
    
    return data


def hilbert_transform(data: np.ndarray, freq_range: Tuple[int, int], fs: int) -> np.ndarray:
    """Calculate Hilbert transform.

    Parameters:
        data - trial data (samples)
        fs - sampling frequency
        freq_range - range of frequencies to calculate gamma for

    Returns:
        gamma - gamma power for each trial (channels X trials)
    """

    # calculate analytic signal
    analytic_signal = hilbert(data)


def calculate_gamma_hilbert(data: np.ndarray, fs: int, freq_range: Tuple[int, int]) -> np.ndarray:
    """Calculate Hilbert transform.

    Parameters:
        data - data from the session (channels X trials X samples)
        fs - sampling frequency
        freq_range - range of frequencies to calculate gamma for

    Returns:
        gamma - gamma power for each trial (channels X trials)
    """
    # calculate gamma for each channel
    gamma = []
    for channel in data: # data is in the shape of channels X trials X samples
        sub_gamma = []
        for trial in channel: # trial is in the shape of trials X samples
            sub_gamma.append(hilbert_transform(trial, freq_range, fs))
        gamma.append(sub_gamma)

    return np.array(gamma)


def load_data_mne(
        data: np.ndarray,
        labels: List[float],
        trial_length: float,
        fs: int,
        notch_filter=None,
        plot=False,
        channels=None,
        ch_removed=None,
        trigger_description=None) -> mne.io.RawArray:
    """Load data into mne.RawArray.

    Parameters:
        data - data from the session (channels X trials x samples)
        labels - labels from the session (onset, duration, description)
        trial_length - length of each trial in seconds
        fs - sampling frequency
        notch_filter - notch filter frequency
        plot - plot the data
        channel_names - names of the channels

    Returns:
        raw - mne.RawArray
    """
    # create mne.RawArray
    if channels is None:
        channels = [f'ch{i}' for i in range(len(data))]
    channel_types = ['ecog' for _ in range(len(data))]
    assert len(channels) == len(channel_types), 'Number of channels and channel types must be equal'

    info = mne.create_info(channels, fs, channel_types)
    raw = mne.io.RawArray(data, info)
    raw.apply_function(lambda x: x * 1e-6) # convert to volts

    # add annotations
    trigger_timing = [label / fs for label in labels]
    trigger_length = [trial_length for _ in labels]
    # map the labels onto conditions. There can be more labels than conditions, repeat the conditions
    if trigger_description is None:
        trigger_description = [CONDITIONS[i % len(CONDITIONS)] for i in range(len(labels))]
    else:
        trigger_description = [trigger_description for i in range(len(labels))]
    # for label in labels:
    annotations = mne.Annotations(trigger_timing, trigger_length, trigger_description)
    raw.set_annotations(annotations)

    if notch_filter:
        raw.notch_filter(notch_filter, trans_bandwidth=3)

    if plot:
        raw.plot(lowpass=90, highpass=.1, block=True)
    
    if not ch_removed:
        ch_removed = raw.info['bads']
    
    raw.drop_channels(ch_removed)

    return raw, ch_removed


# def apply_detrend(*args, **kwargs):
#     """Apply detrend to the data."""
#     y, x, r  = detrend(*args, **kwargs)
#     return y


def create_epochs(mne_data: mne.io.RawArray, prestim, poststim) -> mne.Epochs:
    # filter out the non delay periods (Letter3, DigitSpanWM3)
    events_from_annot, _ = mne.events_from_annotations(mne_data)
    return mne.Epochs(mne_data, events_from_annot, tmin=-prestim, tmax=poststim, preload=True, baseline=(0,0))


def z_score_hilbert_data(active_hilbert_data, control_hilbert_data, plot=False) -> List[np.ndarray]:
    """Z-score hilbert data.

    Apply z-score to the hilbert data. The z-score is calculated using the control trials for each channel and applied to the active trials.

    Parameters:
        active_hilbert_data - hilbert data for active trials. Shape is (freq, trials, channels, samples)
        control_hilbert_data - hilbert data for control trials. Shape is (freq, trials, channels, samples)
        plot - plot the last trial for debugging

    Returns:
        z_scored_hilbert_data - z-scored hilbert data. Shape is (freq, trials, channels, samples)
    """
    # loop over each frequency, find the mean and std for each channel in the control data and z-score the active data
    trial_data = {}

    for freq, (tmp_active_hilbert, tmp_control_hilbert) in enumerate(zip(active_hilbert_data, control_hilbert_data)):
        trial_data[freq] = []
        for i, (active_trial, control_trial) in enumerate(zip(tmp_active_hilbert, tmp_control_hilbert)):
            new_channel_data = {chidx: [] for chidx in range(len(active_trial))}
            for chidx, (active_channel, control_channel) in enumerate(zip(active_trial, control_trial)):
                control_channel_mean = np.mean(control_channel)
                control_channel_std = np.std(control_channel)
                new_channel_data[chidx] = (active_channel - control_channel_mean) / control_channel_std
            data = pd.DataFrame.from_dict(new_channel_data, orient='index').transpose()
            trial_data[freq].append(data)

    # plot the last trial for debugging
    if plot:
        data.plot()
        plt.axhline(y=2, color='r', linestyle='-')
        plt.show(block=True)

    return trial_data


def apply_hilbert_transform_to_epochs(epochs: mne.Epochs) -> List[mne.Epochs]:
    hilbert_data = []
    for bin_epochs in epochs:
        hilbert_data.append(bin_epochs.apply_hilbert(picks='ecog', envelope=True).apply_function(np.abs))
    return hilbert_data


def filter_by_bins(mne_data, freq_range):
    """Filter the dataset by a single frequency bin (120 datasets)."""
    binned_filtered_data = {}
    for freq in range(freq_range[0], freq_range[1] + 1): # 30 - 150 
        cp_data = mne_data.copy()
        binned_filtered_data[freq] = cp_data.filter(freq, freq + 1)

    return binned_filtered_data


def bin_data(data, prestim, poststim, interval):
    binned_epochs = []
    for bin_data in data:
        epoch_data = create_epochs(data[bin_data], prestim, poststim)
        binned_epochs.append(epoch_data)
    
    return binned_epochs
    

if __name__ == '__main__':
    """Cli for gamma analysis.

    Gamma analysis for data collected from the cortiQ device. 
    This script will calculate the gamma power for each interval in the prestimulus and poststimulus period
      for a given trial (target / non target).

    Arguments:
        session - path to session directory
        prestim - prestimulus time in ms
        poststim - poststimulus time in ms
        interval - interval in ms
    
    Example:
        python gamma_analysis.py -s "path/session/" -pre -700 -post 700 -int 100
    """
    import argparse
    import logging

    logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format='(%(threadName)-9s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--session', required=False, default=None, type=str)
    parser.add_argument('-pre', '--prestim', default=0, type=int)
    parser.add_argument('-post', '--poststim', default=4000, type=int)
    # make sure the intervaling works for down to 50ms
    parser.add_argument('-int', '--interval', default=4000, type=int)
    # parser.add_argument('-r', '--relative', default=False, type=bool)
    parser.add_argument('-f', '--freq_range', default=(70, 80), type=tuple)

    args = parser.parse_args()
    interval = args.interval
    poststim = args.poststim
    prestim = args.prestim
    freq_range = args.freq_range

    if args.session is None:
        session = load_experimental_data()

    logging.info(f'\nAnalysis Parameters: \nPrestimulus: {prestim} \nPoststimulus: {poststim} \nInterval: {interval} \n')

    # load cortiQ data
    # bci_data, bci_labels, bci_fs = load_data_bdf(session) # this loads the data from the bdf file but the labels are not correct
    data, labels, fs, channel_names = load_data_dat(session)

    # update intervals to seconds
    poststim = poststim / 1000
    prestim = prestim / 1000
    interval = interval / 1000

    # remove the last few labels *HACK for p01* TODO remove for other subjects
    labels = labels[:-7]

    # grab the delay period for the letter conditions
    control_labels = labels[2::7]

    # grab the delay period for WM trials
    active_labels = labels[6::7]

    # load data into the mne format
    active_mne_data, ch_removed = load_data_mne(data, active_labels, poststim, fs, notch_filter=60, plot=True, channels=channel_names, trigger_description='Active')
    control_mne_data, _ = load_data_mne(data, control_labels, poststim, fs, notch_filter=60, plot=False, channels=channel_names, ch_removed=ch_removed, trigger_description='Control')
    
    # filter the dataset by a single frequency bin by steps in the frequency range
    active_binned_filtered_data = filter_by_bins(active_mne_data, freq_range)
    control_binned_filtered_data = filter_by_bins(control_mne_data, freq_range)

    # create epochs for each trial shape (epochs, channels, samples)
    active_binned_epochs = bin_data(active_binned_filtered_data, prestim, poststim, interval)
    control_binned_epochs = bin_data(control_binned_filtered_data, prestim, poststim, interval)

    # calculate hilbert transform for each frequency bin
    active_hilbert_data = apply_hilbert_transform_to_epochs(active_binned_epochs)
    control_hilbert_data = apply_hilbert_transform_to_epochs(control_binned_epochs)

    # z-score the hilbert data. The will return a list of frequencies with a dataframe for each trial
    z_scored_data = z_score_hilbert_data(active_hilbert_data, control_hilbert_data, plot=True)

    # concat the z-scored data into a single dataframe
    freq_range = [f'{freq}' for freq in range(freq_range[0], freq_range[1] + 1)]

    # create a concat pandas dataframe with the z-scored data using the frequency as the key and the trial number as the index
    concat_data = []
    trial_count = 0
    for i in z_scored_data:
        for j, trial_data in enumerate(z_scored_data[i]):
            trial_count += 1
            trial_data['trial'] = j + 1
            trial_data['freq'] = freq_range[i]
            trial_data.set_index(['freq', 'trial'], inplace=True)
            concat_data.append(trial_data)


    # grab the participant id from the session path
    participant_id = session.split('/')[-1] 

    full_data = pd.concat(concat_data)

    # export the data to a csv
    # full_data.to_csv(f'{participant_id}_data.csv')

    # plot the data trials
    # for i in range(trial_count):
    #     full_data.xs(i + 1, level='trial').plot()
    #     plt.axhline(y=2, color='r', linestyle='-')
    #     plt.show(block=True)

    # plot the data for each frequency with averaged trials
    # for freq in freq_range:
    #     full_data.xs(freq, level='freq').groupby('trial').mean().plot()
    #     plt.axhline(y=2, color='r', linestyle='-')
    #     plt.show(block=True)
    # full_data.plot()
    # plt.show(block=True)

    averaged_trial_data = full_data.groupby('freq').mean()
    # plot a heatmap of the data
    sns.heatmap(averaged_trial_data, cmap ='RdYlGn', linewidths = 0.30, annot = True) 
    plt.show(block=True)


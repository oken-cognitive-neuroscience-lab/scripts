"""Gamma analysis for data collected from the cortiQ device."""
import os

import logging
import argparse
import numpy as np

from typing import List, Tuple

from bcipy.helpers.load import load_experimental_data
from scipy.signal import hilbert

from BCI2kReader import BCI2kReader as bci2k

from bcipy.signal.process.decomposition.psd import (
    power_spectral_density, PSD_TYPE)
from bcipy.signal.process.decomposition import continuous_wavelet_transform
from bcipy.signal.process.filter import Notch

import mne
from mne.io import read_raw_bdf
from mne.viz import plot_compare_evokeds
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

def calculate_cwt(data, fs, freq_range=(50, 80)):
    """Calculate CWT Gamma
    Calculate the amount of gamma using CWT.
    """
    freq = abs((freq_range[1] + freq_range[0]) / 2)
    return continuous_wavelet_transform(data, freq, fs)


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

def calculate_gamma_cwt(data: np.ndarray, fs: int, freq_range=(50, 80)) -> np.ndarray:
    """Calculate gamma power for each trial using CWT.

    Parameters:
        data - data from the session (trials X channels X interval X samples)
        fs - sampling frequency
        trial_length - length of each trial in samples
        freq_range - range of frequencies to calculate gamma for
        relative - calculate relative gamma

    Returns:
        gamma - gamma power for each trial (channels X trials)
    """
    # calculate gamma for each channel
    gamma = []
    for channel in data: # data is in the shape of channels X trials X intervals X samples
        sub_gamma = []
        for trial in channel: # trial is in the shape of trials X intervals X samples
            sub_sub_gamma = []
            for interval in trial:
                sub_sub_gamma.append(calculate_cwt(interval, fs, freq_range))
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
        channels=None) -> mne.io.RawArray:
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
    trigger_description = [CONDITIONS[i % len(CONDITIONS)] for i in range(len(labels))]
    # for label in labels:
    annotations = mne.Annotations(trigger_timing, trigger_length, trigger_description)
    raw.set_annotations(annotations)

    if notch_filter:
        raw.notch_filter(notch_filter, trans_bandwidth=3)

    if plot:
        raw.plot(lowpass=90, highpass=20, block=True)
    
    raw.drop_channels(raw.info['bads'])


    return raw

def create_epochs(mne_data: mne.io.RawArray, prestim, poststim) -> mne.Epochs:
    events_from_annot, _ = mne.events_from_annotations(mne_data)
    return mne.Epochs(mne_data, events_from_annot, tmin=-prestim, tmax=poststim, preload=True)
    

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

    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--session', required=False, default=None, type=str)
    parser.add_argument('-pre', '--prestim', default=100, type=int)
    parser.add_argument('-post', '--poststim', default=4000, type=int)
    parser.add_argument('-int', '--interval', default=4000, type=int)
    # parser.add_argument('-r', '--relative', default=False, type=bool)
    parser.add_argument('-f', '--freq_range', default=(50, 80), type=tuple)

    args = parser.parse_args()
    interval = args.interval
    poststim = args.poststim
    prestim = args.prestim
    freq_range = args.freq_range

    if args.session is None:
        session = load_experimental_data()

    logging.info(f'\nAnalysis Parameters: \nPrestimulus: {prestim} \nPoststimulus: {poststim} \nInterval: {interval} \n')

    # bci_data, bci_labels, bci_fs = load_data_bdf(session) # this loads the data from the bdf file but the labels are not correct
    data, labels, fs, channel_names = load_data_dat(session)
    poststim = poststim / 1000
    prestim = prestim / 1000
    interval = interval / 1000
    # load data into mne

    # remove the last three labels *HACK for p01*
    labels = labels[:-3]
    mne_data = load_data_mne(data, labels, poststim, fs, notch_filter=60, plot=True, channels=channel_names)
    # REmove bad epochs
    # create epochs
    epochs = create_epochs(mne_data, prestim, poststim) # has to has some baseline

    # # GET the average per for each 4 second epoch
    # for epoch in epochs:
    #     # get the average for each epoch across the interval
    #     average_per_epoch = epoch.average()


    # # # # find the mean voltage of all the epochs. 
    # average_epochs = epochs.average()
    # # # # subtract the mean voltage from each epoch
    # epochs = epochs.copy().subtract_evoked(average_epochs)


    filtered_epochs = epochs.copy().filter(20, 90)
    bs1_ep = filtered_epochs['1'].average()
    bs2_ep = filtered_epochs['2'].average()
    bs3_ep = filtered_epochs['3'].average()
    bs4_ep = filtered_epochs['4'].average()
    wm1_ep = filtered_epochs['5'].average()
    wm2_ep = filtered_epochs['6'].average()
    wm3_ep = filtered_epochs['7'].average()
    wm4_ep = filtered_epochs['8'].average()
    # plot_compare_evokeds([bs1_ep, wm1_ep], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')
    # plot_compare_evokeds([bs2_ep, wm2_ep], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')
    # plot_compare_evokeds([bs3_ep, wm3_ep], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')
    # plot_compare_evokeds([bs4_ep, wm4_ep], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')

    
    hilbert_data = epochs.copy().filter(15, 95).apply_hilbert(envelope=False) # 1Hz amp between 20-90Hz
    hilbert_data.info

    breakpoint()

    # amplitude
    bs1 = hilbert_data['1'].apply_function(np.abs).average()
    bs2 = hilbert_data['2'].apply_function(np.abs).average()
    bs3 = hilbert_data['3'].apply_function(np.abs).average()
    bs4 = hilbert_data['4'].apply_function(np.abs).average()
    wm1 = hilbert_data['5'].apply_function(np.abs).average()
    wm2 = hilbert_data['6'].apply_function(np.abs).average()
    wm3 = hilbert_data['7'].apply_function(np.abs).average()
    wm4 = hilbert_data['8'].apply_function(np.abs).average()

    # phase
    # bs1 = hilbert_data['1'].apply_function(np.angle).average()
    # bs2 = hilbert_data['2'].apply_function(np.angle).average()
    # bs3 = hilbert_data['3'].apply_function(np.angle).average()
    # bs4 = hilbert_data['4'].apply_function(np.angle).average()
    # wm1 = hilbert_data['5'].apply_function(np.angle).average()
    # wm2 = hilbert_data['6'].apply_function(np.angle).average()
    # wm3 = hilbert_data['7'].apply_function(np.angle).average()
    # wm4 = hilbert_data['8'].apply_function(np.angle).average()


    # plot_compare_evokeds([bs1, bs2, bs3, bs4, wm1, wm2, wm3, wm4], picks='ecog', colors=['r', 'r', 'r', 'r', 'b', 'b', 'b', 'b'], linestyles=['dashed', 'dashed', 'solid', 'dashed', 'dashed', 'dashed', 'solid', 'dashed'], show=True)
    # plot_compare_evokeds([bs1, wm1], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')
    # plot_compare_evokeds([bs2, wm2], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')
    # plot_compare_evokeds([bs3, wm3], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')
    # plot_compare_evokeds([bs4, wm4], picks='ecog', colors=['r', 'b'], linestyles=['dashed', 'solid'], show=True, combine='median')


    # breakpoint()
    # # data, labels, fs = preprocess_data(data, labels, fs)
    # trials = reshape_data_into_trials(data, labels, poststim, prestim, interval, fs) # Channels X Trials X Intervals X Samples
    # # breakpoint()
    # # calculate gamma for each trial
    # gamma = calculate_gamma(trials, fs, interval, freq_range=freq_range, relative=False)
    # # convert to trials, channels, samples
    # # breakpoint()
    # # trials = np.swapaxes(trials, 0, 1)
    # # gamma_cwt = calculate_gamma_cwt(trials, fs, interval, freq_range=freq_range)
    # # print(gamma_cwt)
    # print(gamma)


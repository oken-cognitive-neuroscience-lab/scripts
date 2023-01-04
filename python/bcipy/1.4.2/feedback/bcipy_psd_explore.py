import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from bcipy.helpers.load import read_data_csv
from bcipy.signal.process.filter import bandpass, notch, downsample
from bcipy.helpers.task import trial_reshaper
from bcipy.helpers.load import load_experimental_data
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition import (
    analysis_channels, analysis_channel_names_by_pos)
from bcipy.signal.process.decomposition.psd import (
    power_spectral_density, PSD_TYPE)

# BciPy Constants
# [TODO] We can load some of these from the session parameter files
MODE = 'calibration'
TRIGGERS_FN = 'triggers.txt'
RAW_DATA_FN = 'raw_data.csv'
CSV_EXPORT_NAME = 'feedback_exports.csv'

# Parameters
TRIAL_LENGTH = 2.5
NUMBER_OF_STIMULI_PER_SEQUENCE = 10
DOWNSAMPLE_RATE = 2
NOTCH_FREQ = 60
FILTER_HP = 2
FILTER_LP = 40

# Quantile Exports
QUANTILES = [15, 30, 45, 70]

# PSD Parameters
"""Define bands here and add to PSD_TO_DETERMINE list."""
ALPHA = ('alpha', [8, 11.99])
ALPHA_SUB_1 = ('alpha_sub_1', [7.00, 9.00])
ALPHA_SUB_2 = ('alpha_sub_2', [11.5, 12.5])
BETA = ('beta', [12, 25])
THETA = ('theta', [4, 7.99])
THETA_SUB_1 = ('theta_sub_1', [3.00, 5.00])
DELTA = ('delta', [1, 3.99])
DELTA_SUB_1 = ('delta_sub_1', [3.20, 4.00])

# append desired psd defined above to the list to use
PSD_TO_DETERMINE = [ALPHA, ALPHA_SUB_1, ALPHA_SUB_2, BETA, THETA, THETA_SUB_1, DELTA]

# Initialize exports
exports = {}
for name, band in PSD_TO_DETERMINE:
        exports[name] = {}
        exports[name]['data'] = []



def psd_explore(
        data_folder,
        channel_index,
        plot=True,
        relative=False,
        reverse=False,
        export_to_csv=False):
    """PSD Explore.

    This assumes use with VR300 for the AD Feedback experiment.

    data_folder: path to a BciPy data folder with raw data and triggers
    channel_index: channel to use for PSD calculation
    plot: whether or not to plot the filtered data and psd spectrum
    relative: whether or not to export relative PSD output
    reverse: whether the level estimations should be descending (default; ie band increases with attention) or ascending
    export_to_csv: whether or not to write output to csv

    returns: average, standard deviation
    """

    # construct the relevant data paths
    trigger_path = f'{data_folder}/{TRIGGERS_FN}'
    raw_data_path = f'{data_folder}/{RAW_DATA_FN}'

    # print helpful information to console
    print('CONFIGURATION:\n'
          f'Trial length: {TRIAL_LENGTH} \n'
          f'Downsample rate: {DOWNSAMPLE_RATE} \n'
          f'Notch Frequency: {NOTCH_FREQ} \n'
          f'Bandpass Range: [{FILTER_HP}-{FILTER_LP}] \n'
          f'Trigger Path: [{trigger_path}] \n'
          f'Raw Data Path: [{raw_data_path}] \n')

    # process and get the data from csv
    raw_data, _, channels, type_amp, fs = read_data_csv(raw_data_path)

    # print helpful information to console
    print(
        'DEVICE INFO:'
        f'\nChannels loaded: {channels}. \n'
        f'Using channel: {channels[channel_index]} \n'
        f'Using Device: {type_amp} - {fs} samples/sec \n')

    # filter the data
    filtered_data, sampling_rate_post_filter = filter_data(
        raw_data, fs, DOWNSAMPLE_RATE, NOTCH_FREQ)

    # decode triggers and get a channel map
    _, trigger_targetness, trigger_timing, offset = trigger_decoder(
        mode=MODE,
        trigger_path=trigger_path)

    # add a static offset of 100 ms [TODO load from parameters]
    offset = offset + .1

    # reshape the data
    x, y, num_seq, _ = trial_reshaper(
        trigger_targetness,
        trigger_timing,
        filtered_data,
        mode=MODE,
        fs=fs,
        k=DOWNSAMPLE_RATE,
        offset=offset,
        channel_map=analysis_channels(channels, type_amp),
        trial_length=TRIAL_LENGTH)

    data = create_sequence_exports(
        x,
        num_seq * 10,
        channel_index,
        TRIAL_LENGTH,
        sampling_rate_post_filter,
        plot,
        relative,
        reverse)

    # plot raw data for the trial index given
    if plot:
        time = np.arange(
            data.size) / sampling_rate_post_filter
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plt.plot(time, data, lw=1.5, color='k')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Voltage')
        plt.xlim([time.min(), time.max()])
        plt.title('Raw Data Plot')
        sns.set(font_scale=1.2)
        sns.despine()
        plt.show()

    if export_to_csv:
        export_data_to_csv(exports)

    return exports


def create_sequence_exports(
        data,
        num_trials,
        channel_index,
        trial_length,
        sampling_rate,
        plot,
        relative,
        reverse,
        step=NUMBER_OF_STIMULI_PER_SEQUENCE):
    """Create Sequence exports.

    Loops through segmented data and calculates the PSD sequence data.

    data: reshaped trial data ['first', 'second']
    num_trials: total number of sequences in task (ie 50, 100)
    channel_index: channel we're interested in extracting
    trial_length: length of reshaping
    sampling_rate: data sampling rate of EEG
    plot: whether or not to plot the data for exploration
    relative: whether this is a relative or absolute calculation of PSD
    reverse: whether the level estimations should be descending (default; ie band increases with attention) or ascending
    step: how many stimuli between each trial [TODO: this could be taken from parameters from the session]
        * we want the PSD from the first stimuli in trial to the trial_length
    """

    index = 0
    frames = int(num_trials / step)
    tmp = []

    # Calculate PSD for every sequence (called frame here)
    for _ in range(frames):
        process_data = data[channel_index][index]
        tmp.append(process_data)
        index += step

        for name, band in PSD_TO_DETERMINE:

            exports[name]['data'].append(
             power_spectral_density(
                process_data,
                band,
                sampling_rate=sampling_rate,
                window_length=TRIAL_LENGTH,
                method=PSD_TYPE.WELCH,
                plot=False,
                relative=relative))

    # calculate the fields of interest for export
    for name, band in PSD_TO_DETERMINE:
        stats_data = np.array(exports[name]['data'])
        exports[name]['average'] = np.mean(stats_data, axis=0)
        exports[name]['stdev'] = np.std(stats_data, axis=0)
        exports[name]['range'] = [
            np.min(stats_data, axis=0), np.max(stats_data, axis=0)
        ]
        if reverse:
            QUANTILES.reverse()
        exports[name]['quantiles'] = np.percentile(stats_data, QUANTILES)
        del exports[name]['data']

    # calculate a raw data average for plotting purposes only
    average = np.mean(np.array(tmp), axis=0)

    if plot:
        power_spectral_density(
                average,
                [1, 2],
                sampling_rate=sampling_rate,
                window_length=TRIAL_LENGTH,
                method=PSD_TYPE.WELCH,
                plot=plot,
                relative=relative)

    return average


def filter_data(raw_data, fs, downsample_rate, notch_filter_freqency):
    """Filter Data.

    Using the same procedure as AD supplement, filter and downsample the data
        for futher processing.

    Return: Filtered data & sampling rate
    """
    notch_filterted_data = notch.notch_filter(
        raw_data, fs, notch_filter_freqency)
    bandpass_filtered_data = bandpass.butter_bandpass_filter(
        notch_filterted_data, FILTER_HP, FILTER_LP, fs, order=2)
    filtered_data = downsample.downsample(
        bandpass_filtered_data, factor=downsample_rate)
    sampling_rate_post_filter = fs / downsample_rate
    return filtered_data, sampling_rate_post_filter


def export_data_to_csv(exports):
    with open(CSV_EXPORT_NAME, 'w') as feedback_file:
        writer = csv.writer(
            feedback_file,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)

        # write headers
        writer.writerow(
            ['',
             'Average',
             'Standard Deviation',
             'Range [min max]',
             f'Quantiles {QUANTILES}'])

        # write PSD data
        for name, _ in PSD_TO_DETERMINE:

            writer.writerow(
                [name,
                 exports[name]['average'],
                 exports[name]['stdev'],
                 exports[name]['range'],
                 exports[name]['quantiles']]
            )


if __name__ == '__main__':
    import argparse

    # Define necessary command line arguments
    parser = argparse.ArgumentParser(description='Explore PSD.')
    parser.add_argument('-channel', '--channel',
                        default=6,
                        type=int,
                        help='channel Index to compute PSD')
    parser.add_argument('-plot', '--plot',
                        default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Whether or not to plot raw data and PSD')
    parser.add_argument('-relative', '--relative',
                        default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Whether or not to use relative band calculation for PSD')
    parser.add_argument('-path', '--path',
                        default=False,
                        type=str,
                        help='Path to BciPy data directory of interest.')
    parser.add_argument('-feedback_desc', '--feedback_desc',
                        default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='By default, PSD are assumed desceding in ' \
                             'nature; ie PSD increases with attention. ' \
                             'Use this flag to reverse that direction. ' \
                             'Used to calculate appropriate cutoffs for feedback levels ')
    parser.add_argument('-export', '--export',
                        required=False,
                        default=False,
                        type=str,
                        help='Path to BciPy data directory of interest.')

    # parse and define the command line arguments.
    args = parser.parse_args()
    data_folder = args.path

    # Note: this doesn't work on Mac for some reason... supply the path in the console
    if not data_folder:
        data_folder = load_experimental_data()

    channel_index = args.channel
    plot = args.plot
    relative_calculation = args.relative
    reverse = args.feedback_desc
    export_to_csv = args.export

    # ignore some pandas warnings, run the psd explore function and print results
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # explore!
        psd = psd_explore(
            data_folder,
            channel_index,
            plot=plot,
            relative=relative_calculation,
            reverse=reverse,
            export_to_csv=export_to_csv)
        print(
            'RESULTS:\n'
            f'{psd}')

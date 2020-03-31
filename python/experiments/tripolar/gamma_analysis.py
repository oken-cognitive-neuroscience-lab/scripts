"""
Transforms needed to do Gamma Analysis on Tripolar Electrodes.

Written By. Tab Memmott, tab.memmott@gmail.com
"""
import csv

from bcipy.helpers.load import read_data_csv, load_json_parameters
from bcipy.signal.process.filter import bandpass, notch, downsample
from bcipy.helpers.task import trial_reshaper
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition import (
    analysis_channels, analysis_channel_names_by_pos)
from bcipy.signal.process.decomposition.psd import (
    power_spectral_density, PSD_TYPE)

# defaults and constants
DOWNSAMPLE_RATE = 2
NOTCH_FREQ = 60
FILTER_HP = 5
FILTER_LP = 90
FILTER_ORDER = 8

# pre-stim / post-stim gamma. target / nontarget. Each individual. r

GAMMA_RANGE = [50, 80]
# EXPORT_CHANNELS = [13, 14, 15, 16]

TRIAL_LENGTH = 100


def calculate_gamma_dwt():
    """Calculate DWT Gamma.

    Calculate the amount of gamma using DWT and plot.
    """
    # TODO
    return


def calculate_fft(data, fs, trial_length, relative=True):
    """Calculate FFT Gamma

    Calculate the amount of gamma using FFT.
    """
    return power_spectral_density(
                data,
                GAMMA_RANGE,
                sampling_rate=fs,
                window_length=trial_length,
                method=PSD_TYPE.WELCH,
                plot=False,
                relative=relative)


def get_experiment_data(raw_data_path, parameters_path, apply_filters=False):
    """Get Experimental Data.

    Given the path to raw data and parameters, parse them into formats we can
        work with. Optionally apply the default filters to the data.
        To change filtering update the constants at the top of file or do custom
        tranforms after the if apply_filters check.
    """
    raw_data, _, channels, type_amp, fs = read_data_csv(raw_data_path)
    parameters = load_json_parameters(parameters_path,
                                      value_cast=True)

    if apply_filters:
        # filter the data as desired here!
        raw_data, fs = filter_data(
            raw_data, fs, DOWNSAMPLE_RATE, NOTCH_FREQ)
    return raw_data, channels, type_amp, fs, parameters


def get_triggers(trigger_path, poststim, prestim=False):
    # decode triggers
    _, trigger_targetness, trigger_timing, offset = trigger_decoder(
        mode='calibration',
        trigger_path=trigger_path)

    # prestim must be a positive number. Transform the trigger timing if
    #   a prestimulus amount is wanted. Factor that into the trial length for
    #   later reshaping
    if prestim and abs(prestim) > 0:
        trigger_timing = transform_trigger_timing(trigger_timing, prestim)
        trial_length = poststim + abs(prestim)
    else:
        trial_length = poststim

    return trigger_timing, trigger_targetness, offset, trial_length


def transform_trigger_timing(trigger_timing, pre_stim):
    """Transform Trigger Timing.

    Given a list of times and a prestimulus amount, shift every
        item in the array by that amount and return the new triggers.

    Note. Given pre_stim is in ms and triggers are in seconds, we 
        transform that here.
    """
    new_triggers = []
    prestim_ms = pre_stim / 1000
    for trigger in trigger_timing:
        new_triggers.append(trigger - prestim_ms)

    return new_triggers


def filter_data(raw_data, fs, downsample_rate, notch_filter_freqency):
    """Filter Data.
    Using the same procedure as AD supplement, filter and downsample the data
        for futher processing.
    Return: Filtered data & sampling rate
    """
    notch_filterted_data = notch.notch_filter(
        raw_data, fs, notch_filter_freqency)
    bandpass_filtered_data = bandpass.butter_bandpass_filter(
        notch_filterted_data, FILTER_HP, FILTER_LP, fs, order=FILTER_ORDER)
    filtered_data = downsample.downsample(
        bandpass_filtered_data, factor=downsample_rate)
    sampling_rate_post_filter = fs / downsample_rate
    return filtered_data, sampling_rate_post_filter


def parse(
        data,
        fs,
        channels,
        type_amp,
        triggers,
        targetness,
        offset,
        trial_length,
        parameters):
    """Parse.

    Using the data collected via BciPy, reshape and return
       the parsed data and labels (target/nontarget)
    """

    # add a static offset of the system.
    # This is calculated on a per machine basis.
    # Reach out if you have questions.
    offset = offset + parameters['static_trigger_offset']

    # reshape the data! *Note* to change the channels you'd like returned
    # from the reshaping, create a custom channel map and pass it into
    # the named arg channel_map. It is a list [0, 1, 0, 1], where 1 is
    # a channel to keep and 0 is to remove. Must be the same length as 
    # channels.
    trials, labels, _, _ = trial_reshaper(
        targetness,
        triggers,
        data,
        mode='calibration',
        fs=fs,
        k=DOWNSAMPLE_RATE,
        offset=offset,
        channel_map=analysis_channels(channels, type_amp),
        trial_length=trial_length)

    return trials, labels


def export_data_to_csv(exports, intervals, targetness):
    """Export Data to CSV.
    
    Given an array of exports and column names, write a csv for processing in other systems
    """
    interval_len = len(intervals)
    CSV_EXPORT_NAME = 'tripolar_analysis.csv'
    with open(CSV_EXPORT_NAME, 'w') as tripolar_export:
        writer = csv.writer(
            tripolar_export,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)

        headers = ['']

        for interval in intervals:
            first = f'neg{abs(interval[0])}' if interval[0] < 0 else interval[0]
            second = f'neg{abs(interval[1])}' if interval[1] < 0 else interval[1]
            headers.append(f'interval_{first}_{second}')
        # write headers
        writer.writerow(headers)

        # write PSD data
        i = 0
        for trial in exports:
            row = [targetness[i]]
            for idx in range(interval_len):
                row.append(exports[trial][idx])
            i += 1
                
            
            writer.writerow(row)

def separate_trials(data, labels):
    """Separate Trials.
    
    Given data [np.array] and labels [0, 1], we want to separate 0, 1 trials and return the data.
    """
    pass

def determine_export_bins(gamma_range, interval):
    """Determine export bins.
    
    assumes gamma range value 1 less than 2.
    """
    # determine the range of our two values 
    diff = abs(gamma_range[0] - gamma_range[1])

    intervals = []

    # start with the smallest number and add interval to it
    j = gamma_range[0]
    for _ in range( int(diff / interval)):
        intervals.append([j, j + interval])
        j += interval

    return intervals


def generate_interval_trials(trials, interval, intervals, fs, trial_length):
    """Generate Interval Trials.
    
    Using the trialed data from the trial reshaper, break the data into interval for export
    """
    channel_index = 14
    export = {}

    # convert interval to ms and calculate samples
    samples_per_interval = int((interval) * fs /2 )
    i = 0
    # loop the channels
    for trial in trials[channel_index]:
        j = 0
        k = samples_per_interval
        z = 0
        export[i] = {}
        for _ in intervals:
            # export[i][z] = trial[j:k]
            export[i][z] = calculate_fft(trial[j:k], fs, interval)

            j = k
            k += samples_per_interval
            z += 1

        i += 1

    return export

        


if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafolder', required=True)
    parser.add_argument('-pre', '--prestim', default=-700, type=int)
    parser.add_argument('-post', '--poststim', default=700, type=int)
    parser.add_argument('-int', '--interval', default=100, type=int)
    args = parser.parse_args()

    # extract relevant args
    data_folder = args.datafolder
    prestim = args.prestim
    poststim = args.poststim
    interval = args.interval

    intervals = determine_export_bins([prestim, poststim], interval)

    raw_data_path = '{}/raw_data.csv'.format(data_folder)
    parameter_path = '{}/parameters.json'.format(data_folder)
    trigger_path = '{}/triggers.txt'.format(data_folder)

    logging.info('Reading information from {}, with prestim=[{}] and poststim=[{}]'.format(
        data_folder, prestim, poststim))

    logging.info('Reading EEG data \n')
    # get the experiment data needed for processing.
    # *Note* Constants for filters at top of file. Set apply_filters to true to use the filters.
    data, channels, type_amp, fs, parameters = get_experiment_data(
        raw_data_path, parameter_path, apply_filters=True)

    logging.info('Reading trigger data \n')
    # give it path and poststimulus length (required). Last, prestimulus length (optional)
    triggers, targetness, offset, trial_length = get_triggers(
        trigger_path, poststim, prestim=prestim)

    logging.info('Parsing into trials \n')
    # parse the data and return the trials and labels to use with DWT
    trials, labels = parse(
        data,
        fs,
        channels,
        type_amp,
        triggers,
        targetness,
        offset,
        trial_length,
        parameters)


    exports = generate_interval_trials(trials, interval, intervals, fs, trial_length)

    # Do your analyses or uncomment next line to use debugger here and see what is returned.

    export_data_to_csv(exports, intervals, targetness)

    # # calculate_fft(trials[0][1], fs, trial_length)

    # logging.info('Complete! \n')


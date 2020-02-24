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
DOWNSAMPLE_RATE = 3
NOTCH_FREQ = 60
FILTER_HP = 2
FILTER_LP = 100
FILTER_ORDER = 2

GAMMA_RANGE = [35, 70]

TRIAL_LENGTH = 100


def calculate_gamma_dwt():
    """Calculate DWT Gamma.

    Calculate the amount of gamma using DWT and plot.
    """
    # TODO
    return


def calculate_fft(data, fs, trial_length, relative=False):
    """Calculate FFT Gamma

    Calculate the amount of gamma using FFT.
    """
    psd = power_spectral_density(
                data,
                GAMMA_RANGE,
                sampling_rate=fs,
                window_length=trial_length,
                method=PSD_TYPE.WELCH,
                plot=True,
                relative=relative)

    return psd


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
    if prestim:
        trigger_timing = transform_trigger_timing(trigger_timing, prestim)
        trial_length = poststim + prestim
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
        notch_filterted_data, FILTER_HP, FILTER_LP, fs, order=2)
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
    trials, labels, num_seq, _ = trial_reshaper(
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


def export_data_to_csv(exports):
    """Export Data to CSV.
    
    Given an array of exports and column names, write a csv for processing in other systems
    """
    CSV_EXPORT_NAME = 'tripolar_analysis.csv'
    with open(CSV_EXPORT_NAME, 'w') as tripolar_export:
        writer = csv.writer(
            tripolar_export,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)

        # # write headers
        # writer.writerow(
        #     ['',
        #      'prestim_1_all',
        #      'prestim_1_all',
        #      'prestim_2_all',
        #      f'Quantiles {QUANTILES}'])

        # # write PSD data
        # for name, _ in PSD_TO_DETERMINE:

        #     writer.writerow(
        #         [name,
        #          exports[name]['average'],
        #          exports[name]['stdev'],
        #          exports[name]['range'],
        #          exports[name]['quantiles']]
        #     )

def separate_trials(data, labels):
    """Separate Trials.
    
    Given data [np.array] and labels [0, 1], we want to separate 0, 1 trials and return the data.
    """
    pass




if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafolder', required=True)
    parser.add_argument('-pre', '--prestim', default=None, type=int)
    parser.add_argument('-post', '--poststim', default=TRIAL_LENGTH, type=int)
    args = parser.parse_args()

    data_folder = args.datafolder
    prestim = args.prestim
    poststim = args.poststim

    # interval = 100
    # pre_stim_gamma = [-200, 500]

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

    # Do your analyses or uncomment next line to use debugger here and see what is returned.
    import pdb;pdb.set_trace()

    # calculate_fft(trials[0][1], fs, trial_length)

    logging.info('Complete! \n')


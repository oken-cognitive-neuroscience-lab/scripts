# Gamma analysis on cortiQ data (cortiQ is a tool for the recording of iEEG data)

## Overview

This is a python package for analyzing gamma band activity in iEEG data. The package is designed to be used with the cortiQ system, but can be used with any data that is in the BCI2000 format/.dat. This script requires the following packages:

* bcipy==2.0.0rc3
* BCI2kReader==0.32.dev0 

## Data

Data is collected from patients coming to OHSU for an awake craniotomy. The data is collected using varied electrodes (grid, strip, depth, etc.) and is recorded using the cortiQ system. The data is output as a .dat file along with a PDF of the recording / placement of the montage. 

The patients were all awake and asked to perform a series of memory tasks. The task is as follows:

    Baseline (16 seconds): 
        Subtasks (all 4 seconds):
            - Prompt patient to listen (no audio)
            - Listen to a series of letters (aaa)
            - Hold in memory and wait (no audio)
            - Repeat letters back in the same order

    AudioWM (16 seconds): 
        Subtasks (all 4 seconds):
            - Prompt patient to listen
            - Listen to a series of numbers (1,6,7)
            - Hold in memory and wait (no audio)
            - Repeat them back in reverse order (7,6,1)



### Notes on the data

#### 04/21/23

Data is loading and client works well. Most of the background on the task has been gathered and documented. PSD is working well, however the cwt is not working as expected. The method errors with a: 

```
conv_shape[-1] += int_psi_scale.size - 1
IndexError: list index out of range
```

I think there is something about the data shape that is wrong... I will need to look into this more.

1. debug cwt method
2. determine filtering pipeline
3. determine decomposition pipeline (fft, cwt, etc.) and parameters (interval, etc.)

#### 04/24/23

- Working MNE implementation with the Hilbert. Need to narrow down processing approach and how to analyze the data... what else will be needed?

#### P01

* Location: L4x5 (grid) and Frontal (Get exact location from PDF/Dr. Oken)
* Only the first 8 tasks are valid, even though there are 10 and some change in the data. Recording was not stopped in time in the OR.

#### 07/15/2023
TODO:

    1. fix epoching to use the appropriate time (ex. seconds or samples depending on the data)
    2. filter out unneeded trials (we want to look at the delay period; 3rd letter, 3rd digit span condition)
    3. Add a linear detrend to the data (in the epoching step) 
        https://nbara.github.io/python-meegkit/auto_examples/example_detrend.html
        https://mailman.science.ru.nl/pipermail/fieldtrip/2013-January/018909.html (a high pass filter will remove trends...)
    4. Interval the data (TODO)
    5. Apply z-score on the interval data (not the full trial after hilbert transform) #STOP here for now and schedule a new meeting
    6. plot the data to reduce things (look at the paper Figure 2)
    7. We are interested in how many SDs the z-scored wm trials are from the baseline trials
    8. Notch filters at all the harmonics (60, 120, 180); not as interested in the surrounding frequencies....


    Removing power line noise: https://www.sciencedirect.com/science/article/pii/S1053811919309474?via%3Dihub
    iEEG-BIDS: https://www.nature.com/articles/s41597-019-0105-7 
    EXAMPLE GAMMA PROCESSING: https://mne.tools/0.23/auto_tutorials/clinical/30_ecog.html 

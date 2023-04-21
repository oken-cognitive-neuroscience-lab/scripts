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

#### P01

* Location: L4x5 (grid) and Frontal (Get exact location from PDF/Dr. Oken)
* Only the first 8 tasks are valid, even though there are 10 and some change in the data. Recording was not stopped in time in the OR.
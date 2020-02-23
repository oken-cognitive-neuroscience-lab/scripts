"""Trigger Time Conversion.

In order to import the triggers into other systems, they must be modified with the offsets
    and trigger timing discrepancies from within the file. Those triggers that are not stimuli must also be removed
    from the file.

Written by. Tab Memmott, tab.memmott@gmail.com
"""
def import_file(path):
    """Import and parse the triggers file"""
    if path.lower().endswith('.txt'):

        # open file and readlines
        with open(path, 'r') as trigger_file:
            content = trigger_file.readlines()
        
        # remove new lines
        parsed_text = [x.strip() for x in content]

        # Take out the offsets
        display_trigger = parsed_text[0].split(' ')[-1]
        daq_trigger = parsed_text[-1].split(' ')[-1]

        stimuli = parsed_text[1:-1]

        return float(display_trigger), float(daq_trigger), stimuli
    raise Exception(f'Incorrect path to triggers given! Must end with .txt. Given=[{path}]')

def correct_timing(offset, stimuli):
    """Correct Timing.
    
    Given the offset and stimuli array, return
        an array of data with stimuli corrected to the offset!
    """
    print('Calculating the correct trigger times in relation to offset... \n')
    new_stimuli = []
    for stim in stimuli:
        # The stimuli are in the form ['stimuli_character informtation time']
        components = stim.split(' ')
        # add the offset to the stimuli time
        new_stimuli_time = float(components[2]) + offset

        # append in the correct form to the text file
        new_stimuli.append([f'{components[0]} {components[1]} {new_stimuli_time}'])

    return new_stimuli


def write_new_trigger_file(data, path, file_name='time_corrected_triggers.txt'):
    """Write New Trigger File.
    
    With and array of data, write the new triggers line by line!

    This assumes an array of arrays of strings. For example,
        [['this'], ['would'], ['work!']]
    """
    new_trigger_path = f'{path}/{file_name}'
    print(f'Writing new trigger file to {new_trigger_path} \n')
    with open(new_trigger_path, 'w') as txt_file:
        for line in data:
            txt_file.write(' '.join(line) + '\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True,
                        help='Path to triggers.txt file')

    args = parser.parse_args()
    path = args.path
    static_offset = 0.1

    print(f'\n****** Starting conversion! ****** \n')

    # Import file and extract the relevant information from it
    display_trigger, daq_trigger, stimuli = import_file(path)
    print(f'File Imported! \nTrigger stimuli occurred at {display_trigger}s for the Display clock and'
          f' {daq_trigger}s for the DAQ clock.')

    # Given the daq starts first, subtract the display and static offset to find
    #  the overall offset from the perspective of the display triggers
    offset = daq_trigger - display_trigger + static_offset
    print(f'Trigger offest=[{offset}] seconds.')

    # # Calculate the new stimuli based on the stimuli and offset
    new_stimuli = correct_timing(offset, stimuli)

    # # construct the directory path to save the triggers to
    root_path = path.replace('/triggers.txt', '')

    # # write the new stimuli as a text file
    write_new_trigger_file(new_stimuli, root_path)
    print(f'\n****** Complete! ****** \n')

    



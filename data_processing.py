import datetime
from datetime import timedelta
import pandas as pd
import numpy as np


"""
The following script defines several functions to process time diaries data and extract sequences. 
In particular, there are 8 functions organised as follows:
    - A first group of functions used for encoding time and activities.
            1. intervals: to encode time intervals.
            2. time_encoding: it generates a dictionary with a map between time and numerical interval.
            3. encoding: to encode categories.
            4. general_what: used if we want to work with general categories of activities. 
            5. align_sequences: to create sequences of the same lenght starting all at the same time.
        These functions are called in the following three, which you will work with. 
    - Functions to model sequences in three different ways (and produce base for network models):
            6. sequences_by_time: just create sequences of the same length considering time intervals.
            7. sequences_by_transitions: sequences are based on transitions between activities.
                Note. All sequences start at the same time.
            8. time_and_transitions: it uses the previous function and encode sequences 
                of heterogeneous activities with time intervals.
        Note. All the functions can take as input the same dataframe, which must have at least the following
                columns named as follows: 'id', 'datetime', 'year', 'what', 'where', 'withw', 'mood'.
"""


def intervals(dataframe, start='05:00:00', end='04:30:00'):
    """
    The function takes as input a dataframe and calculate for each row the time interval
    based on the value in the datetime column. One can choose when to start counting intervals.
    The 'end' parameter must be a time equal to the 'start' time minus 30 minutes. This condition is not evaluated. 
    It returns the same dataframe with a new added column named 'time_interval'.
    The function uses the time_encoding function, which creates a dictionary used for mapping like:
      {'5:00' : 1, '5:30' : 2, ...}
    """

    # Create time to code mapping ('5:00' : 1, '5:30' : 2, ...)
    time_code_mapping = time_encoding(start, end)

    # Filter and copy necessary columns from dataframe
    df_filtered = dataframe[['id', 'datetime', 'year',
                             'what', 'where', 'withw', 'mood']].copy()

    # Replace time intervals with corresponding code values
    df_filtered['time_interval'] = df_filtered['datetime'].dt.time.astype(
        str).replace(time_code_mapping)
    return df_filtered


def time_encoding(start='05:00:00', end='04:30:00'):
    """
    It outputs a dictionary with time as key and numerical assigned interval as value.
    Interval depends on the start and end parameters. 
    """
    # Generate time range with corresponding code values
    start_time = datetime.datetime.strptime(start, '%H:%M:%S')
    end_time = datetime.datetime.strptime(end, '%H:%M:%S') + timedelta(days=1)
    time_range = [(start_time + timedelta(minutes=30*i)).strftime('%H:%M:%S')
                  for i in range((end_time - start_time) // timedelta(minutes=30) + 1)]

    # Create time to code mapping ('5:00' : 1, '5:30' : 2, ...)
    time_code_mapping = {time: code for time, code in zip(
        time_range, range(1, len(time_range)+1))}
    return time_code_mapping  # dict


def encoding(categories='all'):
    """
    Encode categories of activities. 
    If the parameter categories is 'all' (default) it encodes each activity (mostly) differently.
    If it is equal to 'general', then activities are encoded in 6 general categories, which are:
        sleeping, eating, moving, working, studying, free time.
    Otherwise, you can pass a dictionary with your preferred encoding. 
    Note. Values in the dictionary must be string of length 2. 
    Hence, accepted parameters are: 'all', 'general' or a dictionary.
    """
    if categories == 'all':
        codes = {
            'Sleeping': 'SL',                                        # sleeping
            'Rest/nap': 'RN',                                        # rest/nap
            'By car': 'MO',                                          # moving
            'By foot': 'MO',
            'By train': 'MO',
            'By bus': 'MO',
            'By bike': 'MO',
            'By motorbike': 'MO',
            'Work': 'WK',                                            # work
            'Working':'WK',
            'Study': 'ST',                                           # studying
            'Lesson': 'LS',                                          # lesson
            # cult.act.
            'Movie Theater Theater Concert Exhibit ...': 'CL',
            'Reading a book; listening to music': 'CL',
            'Sport': 'SP',                                           # sport
            'Al the phone; in chat WhatsApp': 'SM',                  # smartphone
            'Social media (Facebook Instagram etc.)': 'SM',
            'Social life': 'SC',                                     # social life
            'Coffee break cigarette beer etc.': 'FT',                # free time
            'Watcing Youtube Tv-shows etc.': 'FT',
            'Hobbies': 'FT',
            'Shopping': 'FT',
            'Selfcare': 'CA',                                        # care
            'Housework': 'CA',
            'null': 'OT',                                            # other
            'Other': 'OT',
            'Eating': 'EA',                                          # eating
            'Free Time':'FT',
            'Care':'CA',
            'Moving':'MO',
            'Social media':'SM',
            'Studying': 'ST',
            'Cultural activity':'CL'
        }

    elif categories == 'general':  # 6
        codes = {
            'Sleeping': 'SL',                                        # sleeping
            'Rest/nap': 'SL',                                        # rest/nap
            'By car': 'MO',                                         # moving
            'By foot': 'MO',
            'By train': 'MO',
            'By bus': 'MO',
            'By bike': 'MO',
            'By motorbike': 'MO',
            'Work': 'OT',                                           # work
            'Working':'OT',
            'Study': 'ST',                                           # studying
            'Studying': 'ST',
            'Lesson': 'ST',                                         # lesson
            # cult.act.
            'Movie Theater Theater Concert Exhibit ...': 'FT',
            'Reading a book; listening to music': 'FT',
            'Sport': 'FT',                                           # sport
            'Al the phone; in chat WhatsApp': 'FT',                  # smartphone
            'Social media (Facebook Instagram etc.)': 'FT',
            'Social life': 'FT',                                     # social life
            'Coffee break cigarette beer etc.': 'FT',                # free time
            'Watcing Youtube Tv-shows etc.': 'FT',
            'Hobbies': 'FT',
            'Shopping': 'FT',
            'Selfcare': 'FT',                                       # care
            'Housework': 'FT',
            'null': 'OT',                                            # other
            'Other': 'OT',
            'Eating': 'EA',                                           # eating
            'Moving': 'MO',                                      # for general what
            'Free Time': 'FT',                                   # for general what
            'Care':'FT',
            'Moving':'MO',
            'Social media':'FT',
            'Studying': 'ST',
            'Cultural activity':'FT'
        }

    elif isinstance(categories, dict):
        codes = categories['encoding']

    else:
        raise TypeError(
            "Error: Parameter 'categories' must be a dictionary or a string ['all', 'general'].")

    return codes


def modify_what(dataframe, categories='all'):
    """
    The function takes as input the dataframe and replace values in the 'what' column with general categories. 
    """
    if categories == 'all':
        activity_dict = {
            'Sleeping': 'Sleeping',                                        # sleeping
            'Rest/nap': 'Sleeping',                                        # rest/nap
            'By car': 'Moving',                                            # moving
            'By foot': 'Moving',
            'By train': 'Moving',
            'By bus': 'Moving',
            'By bike': 'Moving',
            'By motorbike': 'Moving',
            'Work': 'Working',                                                # work
            'Study': 'Studying',                                           # studying
            'Lesson': 'Lesson',                                            # lesson
            # cult.act.
            'Movie Theater Theater Concert Exhibit ...': 'Cultural activity',
            'Reading a book; listening to music': 'Cultural activity',
            'Sport': 'Sport',                                              # sport
            'Al the phone; in chat WhatsApp': 'Social media',              # smartphone
            'Social media (Facebook Instagram etc.)': 'Social media',
            'Social life': 'Social life',                                  # social life
            'Coffee break cigarette beer etc.': 'Free Time',               # free time
            'Watcing Youtube Tv-shows etc.': 'Free Time',
            'Hobbies': 'Free Time',
            'Shopping': 'Free Time',
            'Selfcare': 'Care',                                            # care
            'Housework': 'Care',
            'null': 'Other',                                               # other
            'Other': 'Other',
            'Eating': 'Eating'                                             # eating
        }

    elif categories == 'general':
        activity_dict = {
            'Sleeping': 'Sleeping',                                        
            'Rest/nap': 'Sleeping',                                        
            'By car': 'Moving',                                            
            'By foot': 'Moving',
            'By train': 'Moving',
            'By bus': 'Moving',
            'By bike': 'Moving',
            'By motorbike': 'Moving',
            'Work': 'Other',                                            
            'Study': 'Studying',                                          
            'Lesson': 'Studying',                                        
            'Movie Theater Theater Concert Exhibit ...': 'Free Time',
            'Reading a book; listening to music': 'Free Time',
            'Sport': 'Free Time',                                           
            'Al the phone; in chat WhatsApp': 'Free Time',                  
            'Social media (Facebook Instagram etc.)': 'Free Time',
            'Social life': 'Free Time',                                   
            'Coffee break cigarette beer etc.': 'Free Time',          
            'Watcing Youtube Tv-shows etc.': 'Free Time',
            'Hobbies': 'Free Time',
            'Shopping': 'Free Time',
            'Selfcare': 'Free Time',                                  
            'Housework': 'Free Time',
            'null': 'Other',                                           
            'Other': 'Other',
            'Eating': 'Eating'                                             
        }
    
    elif isinstance(categories, dict):
        activity_dict = categories['new_activities']

    else:
        raise TypeError(
            "Error: Parameter 'categories' must be a dictionary or a string ['all', 'general'].")

    newdataframe = dataframe.copy()
    newdataframe['what'] = dataframe['what'].replace(activity_dict)

    return newdataframe


def align_sequences(dataframe, start='05:00:00', end='04:30:00'):
    """
    The function takes as input a dataframe with at least the following columns:
        'id', 'datetime', 'year', 'what', 'where', 'withw', 'mood'.
    It outputs a dataframe with sequences of the same length (48).
    Moreover, it updates 'what' null values with 'Sleeping' in night hours, otherwise 'other'.
    """

    df_filtered = intervals(dataframe, start, end)

    # Group data by id and year (general case: same participant in different years)
    grouped_data = df_filtered.groupby(['id', 'year'])

    # Initialize a new DataFrame to store processed data
    new_dataframe = pd.DataFrame(columns=[
                                 'id', 'datetime', 'year', 'what', 'where', 'withw', 'mood', 'time_interval', 'day', 'sequence_day'])

    # Process each group separately
    for group_key, group_data in grouped_data:
        g = group_data.reset_index(drop=True)
        g['day'] = g['datetime'].dt.date

        # Find indices where time interval is 1 (a new day begins)
        indices = np.where(g['time_interval'] == 1)[0]
        indices = np.append(indices, len(g)-1)

        # Create day of sequence values for each interval (such that intervals of the following day are referred to the previous one)
        g['sequence_day'] = pd.Series(np.repeat(
            g.iloc[indices[:-1], 8].values, np.diff(indices)), index=range(len(g)-1))
        g.iloc[len(g)-1, 9] = g.iloc[len(g)-2, 9]

        # Update 'what' null values with 'Sleeping' based on time thresholds
        time_threshold_1 = datetime.time(20, 30)
        time_threshold_2 = datetime.time(7)
        g['time'] = g['datetime'].dt.time
        g.loc[((g['time'] > time_threshold_1) | (g['time'] < time_threshold_2)) & (
            g['what'] == 'null'), 'what'] = 'Sleeping'
        # Otherwise
        g.loc[g['what'] == 'null', ['what']] = 'Other'
        g = g.drop(columns=['time'])

        # Concatenate processed group with the new DataFrame
        new_dataframe = pd.concat([new_dataframe, g], ignore_index=True)

    # Filter out sequences with length != 48
    new_dataframe_filtered = new_dataframe.groupby(
        ['id', 'sequence_day']).filter(lambda x: len(x['what']) == 48)

    return new_dataframe_filtered


def sequences_by_time_intervals(dataframe, categories='all', start='05:00:00', end='04:30:00'):
    """
    The function takes as input:
    1. the dataframe we want to extract the sequences from. 
       --> Alert! The input dataframe must have at least the following columns: 
            'id', 'datetime', 'year', 'what', 'where', 'withw', 'mood'.
    2. The parameter 'categories' accepts strings 'all' and 'general', otherwise a dictionary mapping between 
        all the activities and the preferred encoding. Values in the dictionary must be string of length 2. 

    Sequences are obtained by taking into account time intervals. Thus, they have all the same length.
    All sequences start at the same time (default 5:00). 
    Each time interval is encoded with a number from 1 (5:00) to 48 (4:30^+1)
    Finally, 'what_code' and 'what_next_code' columns are added, which will be the input for network edgelist. 
    """

    # encoding (default all categories)
    codes = encoding(categories=categories)

    # create sequences of 48 elements with fixed start time
    new_dataframe_filtered = align_sequences(dataframe, start, end)

    # Create a copy of filtered DataFrame for further processing
    final_dataframe = new_dataframe_filtered.copy()

    # Add a new column 'what_code' and replace 'what' values with corresponding codes
    final_dataframe['what_code'] = final_dataframe['what'].replace(codes)

    # Create encoding for the network
    final_dataframe['what_code'] = final_dataframe['time_interval'].astype(
        str) + final_dataframe['what_code'].astype(str)

    # Group data by id and sequence_day
    grouped_data_final = final_dataframe.groupby(['id', 'sequence_day'])

    # Initialize a list to store DataFrame groups
    dataframes_to_concat = []

    # Process each group separately
    for group_key, group_data in grouped_data_final:
        gr = group_data.copy()

        # Create a new column 'what_next_code' with shifted values of 'what_code'
        gr['what_next_code'] = gr['what_code'].shift(-1)

        # Append processed group to the list
        dataframes_to_concat.append(gr)

    # Concatenate all groups into a final DataFrame
    final_dataframe_merged = pd.concat(dataframes_to_concat, ignore_index=True)

    return final_dataframe_merged


def sequences_by_transitions(dataframe, categories='all', start='05:00:00', end='04:30:00'):
    """
    The function extracts sequences of transitions between activities (non-homogeneous).
    Sequences start at the same point in time (default 5:00).
    If categories == 'general', then the column activity 'what' is modified. 
    This transformation is needed since it affects the detection of transition to a new activity. 
    By default the category encoding is done by considering all the activities (and not the general ones). 
    The function outputs a dataframe with two columns ['what_code', 'what_next_code'] from which 
        the network edgelist will be extracted. 
    --> Alert! The input dataframe must have at least the following columns: 
            'id', 'datetime', 'year', 'what', 'where', 'withw', 'mood'.
    """
    # Encoding (default all categories)
    codes = encoding(categories)

    # Create sequences of 48 elements with fixed start time
    df_filtered = align_sequences(dataframe)

    # Modify the activity column
    df_filtered = modify_what(df_filtered, categories)

    df_result = pd.DataFrame(
        columns=['id', 'what', 'start', 'end', 'year', 'sequence_day'])
    grouped = df_filtered.groupby(['id', 'year', 'sequence_day'])

    for group in grouped:
        # Remember id, year, and sequence_day
        id_ = group[0][0]
        year = group[0][1]
        seq_day = group[0][2]

        # Select data in the tuple and sort by datetime
        one_day = group[1].sort_values(['datetime']).reset_index(drop=True)

        # Extract the desired columns from the sorted group
        activity = one_day.iloc[:-1, 3]  # what column = 3
        start = one_day.iloc[:-1, 1]  # datetime column
        end = one_day.iloc[1:, 1]

        # Create a new DataFrame with the extracted data
        mod_df = pd.DataFrame({
            'activity': activity.reset_index(drop=True),
            'time_s': start.reset_index(drop=True),
            'time_t': end.reset_index(drop=True)
        })

        # Fix last activity of the day
        mod_df.loc[len(mod_df)] = [one_day.iloc[-1, 3], one_day.iloc[-1,
                                                                     1], one_day.iloc[-1, 1]+timedelta(minutes=30)]

        # Find indexes where there is a change in the activity carried out
        index_list = mod_df['activity'][mod_df['activity']
                                        != mod_df['activity'].shift()].index.tolist()

        # Utils for iteration
        index_list.append(len(mod_df)-1)

        # Create a list of tuples containing values that will be added to df_result
        tuples_list = [(id_, mod_df.iloc[index_list[i], 0], mod_df.iloc[index_list[i], 1],
                        mod_df.iloc[index_list[i+1], 1], year, seq_day) for i in range(len(index_list)-1)]

        # Convert tuples_list to a DataFrame
        new_data = pd.DataFrame(tuples_list, columns=[
                                'id', 'what', 'start', 'end', 'year', 'sequence_day'])

        # Fix last row
        new_data.iloc[len(new_data)-1,
                      3] = mod_df.iloc[index_list[len(index_list)-1], 2]

        # Append new_data to df_result
        df_result = pd.concat([df_result, new_data], ignore_index=True)

    # Group 'df_result' by 'id' and 'sequence_day'
    grouped_result = df_result.groupby(['id', 'sequence_day'])
    dfs_to_concat = []

    # Iterate over each group in 'grouped_result'
    for group_result in grouped_result:
        # Get the data frame corresponding to the group, reset its index, and make a copy
        gr = group_result[1].reset_index(drop=True).copy()
        # Add a new column 'transition' with values equal to the row index plus 1
        gr['transition'] = gr.index + 1
        # Append the modified data frame to 'dfs_to_concat'
        dfs_to_concat.append(gr)

    # Concatenate all data frames in 'dfs_to_concat' into a new data frame 'df_transitions'
    df_transitions = pd.concat(dfs_to_concat, ignore_index=True)

    # Make a copy of 'df_transitions' and add a new column 'what_code' with values equal to 'what'
    df_with_codes = df_transitions.copy()
    df_with_codes['what_code'] = df_with_codes['what']

    # Replace the values in 'what_code' with the corresponding codes in 'codes' dictionary
    df_with_codes = df_with_codes.replace({'what_code': codes})

    # Combine 'transition' and 'what_code' columns into a new column 'what_code' with string values
    df_with_codes['what_code'] = df_with_codes['transition'].astype(
        str) + df_with_codes['what_code'].astype(str)

    # Group 'df_with_codes' by 'id' and 'sequence_day'
    grouped_codes = df_with_codes.groupby(['id', 'sequence_day'])
    dfs_to_concat = []

    # Iterate over each group in 'grouped_codes'
    for group_codes in grouped_codes:
        # Get the data frame corresponding to the group and make a copy
        gr = group_codes[1].copy()
        # Add a new column 'what_next_code' with values obtained by shifting 'what_code' column
        target_col = gr['what_code'].shift(-1)
        gr['what_next_code'] = target_col
        # Append the modified data frame to 'dfs_to_concat'
        # target_col = gr['what'].shift(-1)
        # gr['what_next'] = target_col
        dfs_to_concat.append(gr)

    # Concatenate all data frames in 'dfs_to_concat' into a new data frame 'df_result_final'
    df_result_final = pd.concat(dfs_to_concat, ignore_index=True)

    return df_result_final


def time_and_transitions(dataframe, categories='all', start='05:00:00', end='04:30:00'):
    """
    The function takes as input:
    1. the dataframe we want to extract the sequences from. 
       --> Alert! The input dataframe must have at least the following columns: 
            'id', 'datetime', 'year', 'what', 'where', 'withw', 'mood'.
    2. The parameter 'categories' accepts strings 'all' and 'general', otherwise a dictionary mapping between 
        all the activities and the preferred encoding. Values in the dictionary must be string of length 2. 
    3. 'start' and 'end' parameters, namely first (1) and last (48) time intervals.
    It outputs a dataframe with added columns ['what_code', 'what_next_code'] in which each activity is encoded
    by considering the time interval. Transitions between activities are also detected.

    """

    # set encoding
    codes = encoding(categories)

    # apply processing to detect transitions between activities
    transitions = sequences_by_transitions(dataframe, categories)

    # we don't need previous encoding
    transitions_copy = transitions.drop(
        columns=['what_code', 'what_next_code'])

    time_dict = time_encoding(start, end)

    # create a time interval column extracted from start column
    transitions_copy['time_interval'] = transitions_copy['start'].dt.time.astype(
        str).replace(time_dict)

    # Combine time interval and encoded 'what' columns
    transitions_copy['what_code'] = transitions_copy['time_interval'].astype(
        str) + transitions_copy['what'].replace(codes)

    grouped = transitions_copy.groupby(['id', 'sequence_day'])
    dfs_to_concat = []

    for _, group in grouped:
        gr = group.copy()
        gr['what_next_code'] = gr['what_code'].shift(-1)
        dfs_to_concat.append(gr)

    time_and_transitions = pd.concat(dfs_to_concat, ignore_index=True)
    return time_and_transitions

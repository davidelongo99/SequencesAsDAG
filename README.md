# Time Diary Preprocessor and Visualizer (Directed Acyclic Graphs)

## Overview

This toolkit is designed to preprocess time diary data and visualize sequences of activities as Directed Acyclic Graphs (DAGs). It is suitable for analyzing and visualizing both individual and collective patterns of time use, with the functionality split across two main modules: `my_network_functions.py` and `simpler_network_1.py`.

## Modules

### 1. Activity Sequence Extraction Methods `data_processing.py`

In this section, we describe three different approaches for processing time diary data to extract sequences of activities. Each approach offers a unique perspective on organizing and analyzing the data.

* Approach 1: Fixed-Length Sequences with Time Intervals

    To begin, we create sequences of fixed length, specifically 48 elements. The primary consideration for this approach is the time intervals. We establish a reference point in time from which we start counting the intervals. By default, the reference point is set at 5 a.m., which represents the first interval of the day. This approach allows us to segment the activities based on the predefined time intervals, providing a consistent framework for analysis.
  
* Approach 2: Transition-Based Sequences

    In the second approach, we focus solely on transitions between activities, without considering the time intervals. Similar to the first approach, all sequences start at a specific time interval. This method emphasizes the movement and changes between different activities, disregarding the duration of each activity. It offers insights into activity patterns and preferences based on the transitions observed.

* Approach 3: Combined Time Intervals and Transition Sequences

    The third approach combines both time intervals and transitions between activities, incorporating the advantages of the previous two methods. This approach enables a comprehensive understanding of the data by considering both the time intervals and the transitions between different states. By integrating these two aspects, we gain insights into both the temporal and sequential dimensions of the activity sequences.

By employing these three approaches, we can explore the time diary data from various angles, revealing different aspects and patterns of human activities. Depending on the research objectives and analytical requirements, one or a combination of these approaches can be used to gain valuable insights from the data.

### 2. `my_network_functions.py` and `simpler_network_1.py`

This module provides a set of functions for creating, manipulating, and visualizing DAGs based on time diary data. Key functionalities include:

- **DAG Construction:** Functions for constructing DAGs from time diary data, with nodes representing activities and edges representing transitions between activities.
- **Visualization:** Utilities for visualizing the DAGs using matplotlib and Plotly, offering both static and interactive visualization options.

 The functions defined in `simpler_network_1.py` complements `my_network_functions.py` by providing streamlined functions for specific tasks such as extracting simpler DAG representative of student's most common daily trajectory. Functions are focused on building simplified or abstracted versions of the DAGs for quick analysis or when dealing with large datasets.

### Dependencies

- Python 3.x
- NetworkX
- Matplotlib
- Plotly
- Pandas


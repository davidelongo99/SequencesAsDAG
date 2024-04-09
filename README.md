# Time Diary Preprocessor and Visualizer (Directed Acyclic Graphs)

## Overview

This toolkit is designed to preprocess time diary data and visualize sequences of activities as Directed Acyclic Graphs (DAGs). It is suitable for analyzing and visualizing both individual and collective patterns of time use, with the functionality split across two main modules: `my_network_functions.py` and `simpler_network_1.py`.

## Modules

### 1. `my_network_functions.py`

This module provides a set of functions for creating, manipulating, and visualizing DAGs based on time diary data. Key functionalities include:

- **DAG Construction:** Functions for constructing DAGs from time diary data, with nodes representing activities and edges representing transitions between activities.
- **Visualization:** Utilities for visualizing the DAGs using matplotlib and Plotly, offering both static and interactive visualization options.
- **Preprocessing Utilities:** Functions to preprocess raw time diary data into a format suitable for DAG construction, including data cleaning and aggregation.

### 2. `simpler_network_1.py`

A simplified version or utility module that complements `my_network_functions.py` by providing streamlined functions for specific tasks such as:

- **Simplified DAG Construction:** Functions focused on building simplified or abstracted versions of the DAGs for quick analysis or when dealing with large datasets.
- **Data Transformation:** Utilities for transforming time diary data into edge lists suitable for network analysis and visualization.

### Dependencies

- Python 3.x
- NetworkX
- Matplotlib
- Plotly
- Pandas


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import simpler_network_1
# import simpler_network_2  working on

VALID_NETWORK_TYPES = {1, 2, 3}

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
            'Eating': 'EA'                                           # eating
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
            'Study': 'ST',                                           # studying
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
            'Moving':'MO',                                      # for general what
            'Free Time':'FT'                                    # for general what
        }

    elif isinstance(categories, dict):
        codes = categories['encoding']

    else:
        raise TypeError(
            "Error: Parameter 'categories' must be a dictionary or a string ['all', 'general'].")

    return codes

def create_network(df,
                   what_column,
                   what_next_column,
                   categories = 'all',
                   node_degree_filter=[0, 0],
                   edge_weight_filter = None,
                   edgelist_to_csv=None,
                   rescale_node_size=True,
                   weekdays=None,
                   random_subject=bool or int,
                   return_color_dict=False,
                   color_dict = None,
                   network_type=None):
    
    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError("Network Type must be one of %r." % VALID_NETWORK_TYPES)
    
    df = df.copy()

    if weekdays == True:
        df['weekday'] = pd.to_datetime(df.sequence_day).dt.day_name()
        df = df.loc[(df['weekday'] != 'Saturday')
                    | (df['weekday'] != 'Sunday')]
    elif weekdays == False:
        df['weekday'] = pd.to_datetime(df.sequence_day).dt.day_name()
        df = df.loc[(df['weekday'] == 'Saturday')
                    | (df['weekday'] == 'Sunday')]

    if type(random_subject) is int:
        df = df.loc[df['id'] == random_subject]
        #print('Subject ID:', random_subject)
    elif random_subject:
        n_id = df['id'].sample().values[0]
        df = df.loc[df['id'] == n_id]
        #print('Subject ID:', n_id)

    # Create edgelist
    #if edge_weight_filter is not None and (random_subject != True or type(random_subject)==int):
    if node_degree_filter is not None and (edge_weight_filter is None or edge_weight_filter == 0):
        if network_type == 1:
            counts = simpler_network_1.preprocess_edgelist(df, threshold=node_degree_filter[0], what_column=what_column, what_next_column=what_next_column)
            # Sort by weight in descending order
            counts.sort_values(by='weight', ascending=False, inplace=True)
            # Extract integer part from the 'source' column
            counts['s_i'] = counts['source'].str.extract('(\d+)').astype('int')
            counts = counts.sort_values(by='s_i')
            # Apply the update_edge function
            counts = simpler_network_1.update_edgelist(counts, threshold=node_degree_filter[1])
            # Fix nodes at level1 
            counts = simpler_network_1.remove_node_level1(counts)
            # Fix nodes at last level
            counts = simpler_network_1.remove_node_last_level(counts)
        # else:
            # counts = simpler_network_2.process_allgroups(df, threshold=node_degree_filter[0], what_column=what_column, what_next_column=what_next_column)
            # # Sort by weight in descending order
            # counts.sort_values(by='weight', ascending=False, inplace=True)
            # # Extract integer part from the 'source' column
            # counts['s_i'] = counts['source'].str.extract('(\d+)').astype('int')
            # counts = counts.sort_values(by='s_i')
            # # Apply the update_edge function
            # counts = simpler_network_2.update_edgelist(counts, threshold=node_degree_filter[1])
            # # Fix nodes at level1 
            # counts = simpler_network_2.remove_node_level1(counts)
            # # Fix nodes at last level
            # counts = simpler_network_2.remove_node_last_level(counts)

            # Create network from edgelist
            G = nx.from_pandas_edgelist(counts, create_using=nx.DiGraph(),
                                    source='source', target='target', edge_attr='weight')

    elif node_degree_filter is None and (edge_weight_filter is not None or edge_weight_filter==0):
        counts = df.groupby([what_column, what_next_column]).size().reset_index(name='weight')
        counts.rename(columns={what_column: 'source', what_next_column: 'target'}, inplace=True)
        if random_subject != True or type(random_subject)==int:
            counts = counts.loc[counts['weight'] > edge_weight_filter]
    
        # Create network from edgelist
        G = nx.from_pandas_edgelist(counts, create_using=nx.DiGraph(),
                                    source='source', target='target', edge_attr='weight')

    ## NODES ATTRIBUTES ##

    # 1 - LABEL
    # Set node label attribute
    nx.set_node_attributes(
        G, {i: i[1:] if len(i) < 4 else i[2:] for i in G.nodes()}, "label")

    # 2 - COLOR
    if categories == 'all' and color_dict is None:
        codes = encoding()
        # Generate a random color for each label
        colors = {label: "#{:06x}".format(random.randint(
            0, 0xFFFFFF)) for label in pd.Series(codes.values()).unique()}
    
    elif categories == 'all' and isinstance(color_dict, dict):
        colors = color_dict
        
    elif categories == 'general':
        colors = {'SL': '#0d09ed',  # blu
                  'MO': '#e509ed',  # violet
                  'OT': '#ed0909',  # red
                  'ST': '#ed6009',  # orange
                  'FT': '#87df5b',  # green
                  'EA': '#edcf09'}  # yellow

    # Set node color attribute
    nx.set_node_attributes(G, {i: colors[i[1:]] if len(
        i) < 4 else colors[i[2:]] for i in G.nodes()}, "color")

    # 3 - SIZE
    # Generate a size dictionary
    size_dict = df.groupby(what_column)[what_column].count().to_dict()
    if rescale_node_size == True:
        # Get the minimum and maximum values in the dictionary
        min_value, max_value = min(size_dict.values()), max(size_dict.values())
        # Define the new minimum and maximum values
        new_min, new_max = 30, 200
        # Rescale the values in the dictionary between new_min and new_max
        rescaled_size = {}
        for key, value in size_dict.items():
            rescaled_value = (value - min_value) / (max_value -
                                                    min_value) * (new_max - new_min) + new_min
            rescaled_size[key] = rescaled_value
        # Set node size attribute based on frequency
        nx.set_node_attributes(G, rescaled_size, 'size')
    else:
        nx.set_node_attributes(G, size_dict, 'size')

    # 4 - INTERVAL
    # Set node interval attribute
    nx.set_node_attributes(
        G, {i: int(i[:1]) if len(i) < 4 else int(i[:2]) for i in G.nodes()}, "interval")

    # SAVE EDGELIST AS CSV
    if edgelist_to_csv is not None:
        counts.to_csv(edgelist_to_csv)

    # Return also colors dict (for comparisons)
    if return_color_dict == True:
        return G, counts, colors

    else:
        return G, counts


def draw_network(G, title='', single_subject=False, figsize=(16, 10)):

    # Calculate the logarithmic scale for edge widths
    edge_weights = [edge_attrs['weight']
                    for _, _, edge_attrs in G.edges(data=True)]
    if len(set(edge_weights)) > 1:
        min_weight = min(edge_weights)
        scaled_weights = np.log(np.array(edge_weights) / min_weight)

        # Normalize the scaled weights to the desired range
        if single_subject == True:
            min_width = 0.3
            max_width = 1.3
        else:
            min_width = 0.01  # Minimum width for the edges
            max_width = 0.9  # Maximum width for the edges

        normalized_weights = (
            (scaled_weights - np.min(scaled_weights)) /
            (np.max(scaled_weights) - np.min(scaled_weights))
        ) * (max_width - min_width) + min_width
    else:
        normalized_weights = [0.3] * len(edge_weights)

    # Calculate positions for the nodes
    pos = {}
    for node in G.nodes:
        interval = G.nodes[node]['interval']
        # Calculate y-coordinate based on n_i
        nodes_in_interval = [
            n for n in G.nodes if G.nodes[n]['interval'] == interval]
        n_i = len(nodes_in_interval)
        index = nodes_in_interval.index(node)
        y_coord = (index - (n_i - 1) / 2) * 0.5
        pos[node] = (interval, y_coord)

    # Draw nodes
    node_colors = [node_attrs['color'] for _, node_attrs in G.nodes(data=True)]
    node_sizes = [node_attrs['size'] for _, node_attrs in G.nodes(data=True)]

    # DEFINE FIGURE SIZE
    plt.figure(figsize=figsize)

    # DRAW THE NETWORK
    nx.draw(G, pos, node_color=node_colors,
            node_size=node_sizes, width=normalized_weights)

    # LEGEND
    # Extract node colors for legend
    node_data = G.nodes.data()
    label_color_dict = {}
    unique_labels = set()

    for node, data in node_data:
        label = data['label']
        color = data['color']

        if label not in unique_labels:
            label_color_dict[label] = color
            unique_labels.add(label)
    node_patches = []
    for label, color in label_color_dict.items():
        node_patches.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=node_patches, title='Labels',
               bbox_to_anchor=(1.02, 1), loc='best')

    # TITLE
    plt.title(title)

    plt.show()


def draw_all_networks(network_list, single_subject=False, titles=['Time Intervals', 'Transitions', 'Time and Transitions'], figsize=(10, 15)):
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    ax123 = [axes[0], axes[1], axes[2]]

    for G, ax, title in zip(network_list, ax123, titles):
        # Calculate the logarithmic scale for edge widths
        edge_weights = [edge_attrs['weight']
                        for _, _, edge_attrs in G.edges(data=True)]
        
        if len(set(edge_weights)) > 1:
            min_weight = min(edge_weights)
            scaled_weights = np.log(np.array(edge_weights) / min_weight)

            # Normalize the scaled weights to the desired range
            if single_subject == True:
                min_width = 0.3
                max_width = 1.3
            else:
                min_width = 0.01  # Minimum width for the edges
                max_width = 0.9  # Maximum width for the edges

            normalized_weights = (
                (scaled_weights - np.min(scaled_weights)) /
                (np.max(scaled_weights) - np.min(scaled_weights))
            ) * (max_width - min_width) + min_width
        else:
            normalized_weights = [0.3] * len(edge_weights)

        # Calculate positions for the nodes
        pos = {}
        for node in G.nodes:
            interval = G.nodes[node]['interval']
            # Calculate y-coordinate based on n_i
            nodes_in_interval = [
                n for n in G.nodes if G.nodes[n]['interval'] == interval]
            n_i = len(nodes_in_interval)
            index = nodes_in_interval.index(node)
            y_coord = (index - (n_i - 1) / 2) * 0.5
            pos[node] = (interval, y_coord)

        # Draw nodes
        node_colors = [node_attrs['color']
                       for _, node_attrs in G.nodes(data=True)]
        node_sizes = [node_attrs['size']
                      for _, node_attrs in G.nodes(data=True)]

        # DRAW THE NETWORK
        nx.draw(G, pos, node_color=node_colors,
                node_size=node_sizes, width=normalized_weights, ax=ax)

        # LEGEND
        # Extract node colors for legend
        node_data = G.nodes.data()
        label_color_dict = {}
        unique_labels = set()

        for node, data in node_data:
            label = data['label']
            color = data['color']

            if label not in unique_labels:
                label_color_dict[label] = color
                unique_labels.add(label)

        node_patches = []
        for label, color in label_color_dict.items():
            node_patches.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=node_patches, title='Labels',
                  bbox_to_anchor=(1.02, 1), loc='upper center')

        # TITLE
        ax.set_title(title)

    plt.show()


def draw_network_plotly(G, title=''):

    # Calculate positions for the nodes
    pos = {}
    for node in G.nodes:
        interval = G.nodes[node]['interval']
        # Calculate y-coordinate based on n_i
        nodes_in_interval = [
            n for n in G.nodes if G.nodes[n]['interval'] == interval]
        n_i = len(nodes_in_interval)
        index = nodes_in_interval.index(node)
        y_coord = (index - (n_i - 1) / 2) * 0.5
        pos[node] = (interval, y_coord)

    # Create a Plotly figure
    fig = go.Figure()

    # Add edges to the figure
    edge_trace = go.Scatter(
        x=[pos[edge[0]][0] for edge in G.edges()],
        y=[pos[edge[0]][1] for edge in G.edges()],
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    fig.add_trace(edge_trace)

    # Add nodes to the figure
    node_colors = [node_attrs['color'] for _, node_attrs in G.nodes(data=True)]
    node_sizes = [node_attrs['size']/4 for _, node_attrs in G.nodes(data=True)]
    node_labels = [node_attrs['label'] for _, node_attrs in G.nodes(data=True)]
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers',
        marker=dict(
            color=node_colors,
            size=node_sizes
        ),
        text=node_labels,
        hoverinfo='text'
    )
    fig.add_trace(node_trace)

    # Define layout options
    fig.update_layout(
        title=title,
        showlegend=False,
        legend=dict(
            title='Labels',
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            tickformat=".1f"  # Format the tick labels to one decimal place
        )
    )

    # Show the plot
    fig.show()


# ALTERNATIVE - THIS WORKS
# def draw_network(G):
#     # Calculate positions for the nodes
#     pos = {}
#     for node in G.nodes:
#         interval = G.nodes[node]['interval']
#         # Calculate y-coordinate based on n_i
#         nodes_in_interval = [n for n in G.nodes if G.nodes[n]['interval'] == interval]
#         n_i = len(nodes_in_interval)
#         index = nodes_in_interval.index(node)
#         y_coord = (index - (n_i - 1) / 2) * 0.5
#         pos[node] = (interval, y_coord)

#     plt.figure(figsize=(16, 10))

#     nx.draw(G, pos, node_color=[node_attrs['color'] for _, node_attrs in G.nodes(data=True)],
#             node_size=[node_attrs['size'] for _, node_attrs in G.nodes(data=True)],
#             width=[edge_attrs['weight']/1500 for _, _, edge_attrs in G.edges(data=True)])

#     plt.show()

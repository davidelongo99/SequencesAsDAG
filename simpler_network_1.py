import pandas as pd
import re

"""
Daily Activity Graph Simplification

Description:
This module provides functionality to process and manipulate a Directed Acyclic Graph (DAG) 
representing daily activity sequences carried out by an individual over a span of two weeks. 
These activities were monitored every 30 minutes, resulting in 48 time intervals. Each node 
in the graph represents an activity within a specific time interval.

Key Features:
- Each node (except the first and the last) has equal in-degree and out-degree, where the 
  in-degree represents the number of times an activity was observed, and the out-degree 
  represents the number of times that activity was followed by another.
- Every connection between nodes (edge) has a weight that indicates how many times a particular 
  transition between two activities was observed.

Objective:
To simplify the graph by eliminating nodes that represent less frequent activities (e.g., 
activities with both in-degree and out-degree equal to 1), while still maintaining an accurate 
representation of the individual's primary habits. This translates to identifying and removing 
less significant nodes, redistributing their weights to maintain the graph's coherence.

Functions:
- remove_node: A function that plays a pivotal role in the simplification process. When a node 
  is removed:
  1. The module identifies the most frequent node in the subsequent time interval.
  2. If there already exists a connection between the "source" node and this most frequent node, 
     the weight of the removed connection is added to the weight of the existing connection.
  3. If no such connection exists, a new one is created with the weight of the removed connection.

This function ensures that, while a node is removed, the information regarding transitions 
between activities is preserved as much as possible, reflecting the importance of maintaining 
an accurate representation of the individual's daily habits.
"""

# Function to extract number from a string
def extract_number(s):
    return int(re.findall(r'(\d+)', s)[0])

def find_isolated_nodes(counts, t):
    # Calculate the weighted in-degree for each node
    in_deg = counts.groupby('target').weight.sum()
    out_deg= counts.groupby('source').weight.sum()

    # Find nodes with in-degree == t (which will also have outdegree t)
    nodes_in_deg_t = in_deg[in_deg == t].index.tolist()

    # Find nodes with out-degree == t (which will also have indegree t)
    nodes_out_deg_t = out_deg[out_deg == t].index.tolist()

    nodes = list(set(nodes_in_deg_t).intersection(set(nodes_out_deg_t)))

    # Sort the list by the extracted numbers
    nodes = sorted(nodes, key=extract_number)

    # Add code to remove node with highest weight
    if t == 8:
        to_not_include = []        
        for node in nodes[:]:
            interval = extract_number(node)
            nodes_in_interval = [n for n in nodes if extract_number(n) == interval]
            max_weight_node = max(nodes_in_interval, key=lambda n: in_deg[n])
            if node == max_weight_node:
                to_not_include.append(node)
        nodes = [node for node in nodes if node not in to_not_include]

    return nodes

def remove_node(counts, node, t):
    # interval
    s_i = int(re.findall(r'(\d+)', node)[0])

    # List to collect indices of rows that need to be deleted
    rows_to_drop = []

    # ----- Incoming Section -----
    to_remove_all_incoming = counts.loc[counts['target'] == node].sort_values(by='weight')
    degree = to_remove_all_incoming['weight'].sum()

    to_remove_all_outgoing = counts.loc[counts['source'] == node]
    already_used = []

    for index, to_remove in to_remove_all_incoming.iterrows():
        source = to_remove['source']
        weight = to_remove['weight']

        # Find the row with the max weight for the current source
        max_weight = counts.loc[(counts['s_i'] == s_i-1) & (counts['target']!= node)].sort_values(by='weight', ascending=False).iloc[0]
        new_target = max_weight['target']

        existing_connection = counts.loc[(counts['source'] == source) & (counts['target'] == new_target)]

        if len(existing_connection) > 0:
            existing_connection = existing_connection.iloc[0]
            counts.loc[existing_connection.name, 'weight'] += weight

        else:
            # Create a new DataFrame for the row to be added
            new_row = pd.DataFrame([[source, new_target, weight, s_i-1]], columns=counts.columns, index=[counts.index.max() + 1 ])
            # Concatenate the existing DataFrame with the new row
            counts = pd.concat([counts, new_row])

        # ----- Outgoing Section -----
        # base case in which there is a connection outgoing with the same weight as the incoming one
        outgoing_connections = counts.loc[(counts['source'] == node) & (counts['weight'] == weight)]
        if len(outgoing_connections) > 0:
            for i, outgoing_connection in outgoing_connections.iterrows():
                target = outgoing_connection['target']
                if target in already_used:
                    continue
                else:
                    already_used.append(target)
                    break
            existing_connection = counts.loc[(counts['source'] == new_target) & (counts['target'] == target)]

            if len(existing_connection) != 0:
                existing_connection = existing_connection.iloc[0]
                counts.loc[(counts['source'] == new_target) & (counts['target'] == target), 'weight'] += weight
            else:
                # Create a new DataFrame for the row to be added
                new_row = pd.DataFrame([[new_target, target, weight, s_i]], columns=counts.columns, index=[counts.index.max() + 1 ])
                # Concatenate the existing DataFrame with the new row
                counts = pd.concat([counts, new_row])

            if outgoing_connection.name not in rows_to_drop:
                counts.drop(outgoing_connection.name, inplace=True)

        else:
            outgoing_connections = counts.loc[(counts['source'] == node)].sort_values(by='weight', ascending=False)
            outgoing_weights = outgoing_connections['weight']

            if any(x > weight for x in outgoing_weights.unique()):
                # CASE 2
                max_weight = outgoing_weights.max()
                outgoing_connection = counts.loc[(counts['source'] == node) & (counts['weight'] == max_weight)]

                if len(outgoing_connection) != 0:
                    outgoing_connection = outgoing_connection.iloc[0]
                    target = outgoing_connection['target']

                    counts.loc[outgoing_connection.name, 'weight'] -= weight
                    existing_connection = counts.loc[(counts['source'] == new_target) & (counts['target'] == target)]
                    if len(existing_connection) != 0:
                        existing_connection = existing_connection.iloc[0]
                        counts.loc[(counts['source'] == new_target) & (counts['target'] == target), 'weight'] += weight
                    else:
                        # Create a new DataFrame for the row to be added
                        new_row = pd.DataFrame([[new_target, target, weight, s_i]], columns=counts.columns, index=[counts.index.max() + 1 ])
                        # Concatenate the existing DataFrame with the new row
                        counts = pd.concat([counts, new_row])
                    

            # CASE 1
            else:
                if outgoing_weights.sum() == weight:
                    for index, outgoing_connection in outgoing_connections.iterrows():
                        out_weight = outgoing_connection['weight']
                        target = outgoing_connection['target']

                        existing_connection = counts.loc[(counts['source'] == new_target) & (counts['target'] == target)]
                        if len(existing_connection) != 0:
                            existing_connection = existing_connection.iloc[0]
                            counts.loc[(counts['source'] == new_target) & (counts['target'] == target), 'weight'] += out_weight
                        else:
                            new_row = pd.DataFrame([[new_target, target, out_weight, s_i]], columns=counts.columns, index=[counts.index.max() + 1 ])
                            counts = pd.concat([counts, new_row])

                        if outgoing_connection.name not in rows_to_drop:
                            rows_to_drop.append(outgoing_connection.name)

                else:
                    stop = []
                    for index, outgoing_connection in outgoing_connections.iterrows():
                        if sum(stop) == weight:
                            break
                        else:
                            out_weight = outgoing_connection['weight']
                            target = outgoing_connection['target']

                            existing_connection = counts.loc[(counts['source'] == new_target) & (counts['target'] == target)]
                            if len(existing_connection) != 0:
                                existing_connection = existing_connection.iloc[0]
                                counts.loc[(counts['source'] == new_target) & (counts['target'] == target), 'weight'] += out_weight
                            else:
                                # Create a new DataFrame for the row to be added
                                new_row = pd.DataFrame([[new_target, target, weight, s_i]], columns=counts.columns, index=[counts.index.max() + 1 ])
                                # Concatenate the existing DataFrame with the new row
                                counts = pd.concat([counts, new_row])

                            if outgoing_connection.name not in rows_to_drop:
                                rows_to_drop.append(outgoing_connection.name)

                            stop.append(out_weight)

        # Mark the row for deletion
        rows_to_drop.append(to_remove.name)

    # Drop rows marked for deletion in the incoming section
    counts.drop(rows_to_drop, inplace=True)
    counts.reset_index(drop=True, inplace=True)
    return counts


def check_in_out(counts, return_df=False):
    # Calculate in-degrees and out-degrees
    in_deg = counts.groupby('target')['weight'].sum()
    out_deg = counts.groupby('source')['weight'].sum()

    # Initialize an empty list to store nodes with different in-degree and out-degree
    nodes_with_mismatch = []

    # Iterate through unique nodes and check for differences in in-degree and out-degree
    for node in in_deg.index.unique():
        if node in out_deg.index.unique():
            if in_deg.get(node, 0) != out_deg.get(node, 0):
                nodes_with_mismatch.append({
                    'node': node,
                    'in_degree': in_deg.get(node, 0),
                    'out_degree': out_deg.get(node, 0)
                })

    # Create a dataframe from the list of nodes with different in-degree and out-degree
    nodes_df = pd.DataFrame(nodes_with_mismatch)
    
    if return_df:
        return nodes_df
    else:
        if len(nodes_df) == 0:
            return True
        else:
            return False
        

def update_edgelist(counts, threshold):
    for t in range(1, threshold+1):
        nodes = find_isolated_nodes(counts, t)
        while len(nodes) != 0:
            node = nodes[0]
            counts = remove_node(counts, node, t)
            nodes = find_isolated_nodes(counts, t)
            # if check_in_out(counts) == False:
            #     print(check_in_out(counts, return_df=True))
            #     raise Exception("Handling some problems")
    return counts
    
def process_group(grp, threshold, what_column, what_next_column):
    # Create edgelist
    counts = grp.groupby([what_column, what_next_column]).size().reset_index(name='weight')
    counts.rename(columns={what_column: 'source', what_next_column: 'target'}, inplace=True)

    # Sort by weight in descending order
    counts.sort_values(by='weight', ascending=False, inplace=True)

    # Extract integer part from the 'source' column
    counts['s_i'] = counts['source'].str.extract('(\d+)').astype('int')

    counts = counts.sort_values(by='s_i')

    # Apply the update_edge function
    counts = update_edgelist(counts, threshold)

    # Fix nodes at level1 
    counts = remove_node_level1(counts)

    # Fix nodes at last level
    counts = remove_node_last_level(counts)

    return counts


def preprocess_edgelist(df, threshold, what_column, what_next_column):
    # Group by 'id'
    grpd = df.groupby('id')

    # Check if the threshold is within the valid range
    if 0 <= threshold <= 10:
        new_edgelist = []

        # Process each group
        for _, grp in grpd:
            #print(_)
            new_edgelist.append(process_group(grp, threshold, what_column, what_next_column))

        new_edgelist = pd.concat(new_edgelist)
        grouped = new_edgelist.groupby(['source', 'target'], as_index=False)['weight'].sum()

        return grouped
    else:
        raise ValueError("Error: The threshold must be between 0 and 10.")
    

def remove_node_level1(edgelist1):
    edgelist1['s_i'] = edgelist1['source'].apply(extract_number)
    filtered_rows = edgelist1.loc[edgelist1['s_i'] == 1]

    max_weight_row = filtered_rows.sort_values(by='weight', ascending=False).iloc[0]
    new_source = max_weight_row['source']

    for index, row in filtered_rows.iterrows():
        if index == max_weight_row.name:
            pass
        else:
            weight = row['weight']
            target = row['target']

            existing_connection = edgelist1.loc[(edgelist1['source'] == new_source) & (edgelist1['target'] == target)]

            if len(existing_connection) > 0:
                existing_connection = existing_connection.iloc[0]
                edgelist1.loc[existing_connection.name, 'weight'] += weight

            else:
                # Create a new DataFrame for the row to be added
                new_row = pd.DataFrame([[new_source, target, weight, 1]], columns=edgelist1.columns, index=[edgelist1.index.max() + 1 ])
                # Concatenate the existing DataFrame with the new row
                edgelist1 = pd.concat([edgelist1, new_row])
            
            edgelist1.drop(index, inplace=True)
    
    edgelist1 = edgelist1.drop(columns=['s_i'])
            
    return edgelist1

def remove_node_last_level(edgelist1):
    edgelist1['s_i'] = edgelist1['target'].apply(extract_number)
    max_s_i = edgelist1['s_i'].max()
    filtered_rows = edgelist1.loc[edgelist1['s_i'] == max_s_i]

    max_weight_row = filtered_rows.sort_values(by='weight', ascending=False).iloc[0]
    new_target = max_weight_row['target']

    for index, row in filtered_rows.iterrows():
        if index == max_weight_row.name:
            pass
        else:
            weight = row['weight']
            source = row['source']

            existing_connection = edgelist1.loc[(edgelist1['source'] == source) & (edgelist1['target'] == new_target)]

            if len(existing_connection) > 0:
                existing_connection = existing_connection.iloc[0]
                edgelist1.loc[existing_connection.name, 'weight'] += weight

            else:
                # Create a new DataFrame for the row to be added
                new_row = pd.DataFrame([[source, new_target, weight, max_s_i]], columns=edgelist1.columns, index=[edgelist1.index.max() + 1 ])
                # Concatenate the existing DataFrame with the new row
                edgelist1 = pd.concat([edgelist1, new_row])
            
            edgelist1.drop(index, inplace=True)
    
    edgelist1 = edgelist1.drop(columns=['s_i'])
    return edgelist1

######### EXAMPLE USAGE ###########
# import simpler_network

# # Load the CSV file
# time_intervals = pd.read_csv('time_intervals.csv')
# what_column = 'what_code'
# what_next_column = 'what_next_code'

# counts = preprocess_edgelist(df, threshold=edge_weight_filter, what_column=what_column, what_next_column=what_next_column)


    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 11:50:12 2025

@author: georgieauvray
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os
import warnings
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

# INITIAL DATA LOADING
# Folder containing all CSV files
folder_path = "csv"
# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
# Load all CSVs into a list of DataFrames
dataframes = [pd.read_csv(file) for file in csv_files]
# Optionally, combine them into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# CREATING A DICTIONARY OF GROUPS
combined_df['Possession'] = combined_df['teamName'].shift(-1)
grouped_dict = {}
group_counter = 0  # Unique key for each group

for fxid, group in combined_df.groupby(['FXID', 'SetNum']):
    group = group.sort_values('ID')  # Ensure order by ID

    current_group = []
    prev_playnum = None  

    for _, row in group.iterrows():
        if prev_playnum is not None and row['PlayNum'] != prev_playnum:
            # Store the completed group with a numeric key
            grouped_dict[group_counter] = pd.DataFrame(current_group)
            group_counter += 1  # Increment the counter
            current_group = []  # Reset for the new group
        
        current_group.append(row)
        prev_playnum = row['PlayNum']  # Update PlayNum tracker

    # Store the last collected group (if any)
    if current_group:
        grouped_dict[group_counter] = pd.DataFrame(current_group)
        group_counter += 1  # Increment the counter

def create_grouped_dataset_with_setnum_try(grouped_dict):
    grouped_data = []
    
    # Identify which SetNum & FXID combinations contain a Try
    try_setnums = set(
        (group['FXID'].iloc[0], group['SetNum'].iloc[0])
        for group in grouped_dict.values() 
        if (group['actionName'] == 'Try').any()
    )

    for group_id, group in grouped_dict.items():
        # Remove groups that contain actionName == 'Goal Kick'
        if (group['actionName'] == 'Goal Kick').any():
            continue  

        first_row = group.iloc[0]
        last_row = group[group['x_coord_end'] != 0].iloc[-1] if any(group['x_coord_end'] != 0) else first_row
        last_row2 = group.iloc[-1]

        # Assign TryScored based on SetNum & FXID
        try_scored = int((first_row['FXID'], first_row['SetNum']) in try_setnums)

        # Count passes and carries
        num_passes = (group['actionName'] == 'Pass').sum()
        num_carries = (group['actionName'] == 'Carry').sum()

        # Compute total distance (sum of segment distances)
        valid_moves = group[group['x_coord_end'] != 0]
        total_distance = np.sum(
            np.sqrt((valid_moves['x_coord_end'] - valid_moves['x_coord'])**2 +
                    (valid_moves['y_coord_end'] - valid_moves['y_coord'])**2)
        )

        # Compute direct distance (first to last valid point)
        direct_distance = np.sqrt(
            (last_row['x_coord_end'] - first_row['x_coord'])**2 +
            (last_row['y_coord_end'] - first_row['y_coord'])**2
        )

        # Compute path area (Shoelace formula)
        path_area = 0
        if len(valid_moves) > 1:
            x_vals = valid_moves['x_coord'].tolist() + [last_row['x_coord_end']]
            y_vals = valid_moves['y_coord'].tolist() + [last_row['y_coord_end']]
            path_area = 0.5 * abs(
                sum(x_vals[i] * y_vals[i+1] - x_vals[i+1] * y_vals[i] for i in range(len(x_vals) - 1))
            )
        path_area_root = np.sqrt(path_area)
        
        # Compute directness
        directness = 0 if direct_distance == 0 else total_distance / direct_distance

        grouped_data.append({
            'GroupID': group_id,  
            'FXID': first_row['FXID'],  
            'SetNum': first_row['SetNum'],  
            'PlayNum': first_row['PlayNum'],  
            'x_coord': first_row['x_coord'],
            'y_coord': first_row['y_coord'],
            'x_coord_end': last_row['x_coord_end'],
            'y_coord_end': last_row['y_coord_end'],
            'teamName': first_row['teamName'],
            'sequence_id': first_row['sequence_id'],
            'TryScored': try_scored,
            'Possession': last_row2['Possession'],
            'num_passes': num_passes,
            'num_carries': num_carries,
            'total_distance': total_distance,
            'direct_distance': direct_distance,
            'path_area_root': path_area_root,
            'directness': directness
        })
    
    return pd.DataFrame(grouped_data)

grouped_set_df = create_grouped_dataset_with_setnum_try(grouped_dict)

## Regression model
# Define X and y
X = grouped_set_df[['x_coord', 'x_coord_end', 'directness', 'num_carries', 'path_area_root']]
X['y_coord_diff'] = np.abs(grouped_set_df['y_coord_end'] - grouped_set_df['y_coord'])
X['y_coord_diff_2'] = X['y_coord_diff'] ** 2
X['directness_x_distance'] = X['directness'] * (X['x_coord_end'] - X['x_coord'])
X['x_coord_end_x_directness'] = X['x_coord_end'] * X['directness']
X = X.drop('x_coord_end', axis=1)

y = grouped_set_df['TryScored']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add constant to train and test sets
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit model
xT_model = sm.Logit(y_train, X_train).fit()

# Predict probabilities on test set
y_pred_xT = xT_model.predict(X_test)

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_xT)
roc_auc = auc(fpr, tpr)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
fig.set_facecolor('whitesmoke')

ax.plot(fpr, tpr, color='navy', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontproperties='serif')
ax.set_ylabel('True Positive Rate', fontproperties='serif')
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontproperties='serif', fontsize=14)
ax.legend(loc="lower right", prop={'family': 'serif'})

plt.tight_layout()
plt.show()


## LOOKING AT PLAYERS
def calculate_xT(x_coord, y_coord, x_coord_end, y_coord_end, num_carries, directness, path_area_root, model):
    # Calculate additional features
    y_coord_diff = abs(y_coord_end - y_coord)
    y_coord_diff_2 = y_coord_diff ** 2
    directness_x_distance = directness * (x_coord_end - x_coord)
    x_coord_end_x_directness = x_coord_end * directness
    
    # Create the input DataFrame with all features
    X_new = pd.DataFrame({
        'const': [1],  # For the intercept
        'x_coord': [x_coord],
        'directness': [directness],
        'num_carries': [num_carries],
        'path_area_root': [path_area_root],
        'y_coord_diff': [y_coord_diff],
        'y_coord_diff_2': [y_coord_diff_2],
        'directness_x_distance': [directness_x_distance],
        'x_coord_end_x_directness': [x_coord_end_x_directness]
    })

    # Predict xT using the model
    xT = model.predict(X_new).item()
    return xT


def player_xT(playerName, actions, group, model):
    pass_group = group[group['playerName'] == playerName]
    pass_action = pass_group[pass_group['actionName'].isin(actions)]
    
    if pass_action.empty:
        return None
    
    first_row = group.iloc[0]
    last_row = group[group['x_coord_end'] != 0].iloc[-1] if any(group['x_coord_end'] != 0) else first_row
    last_row2 = group.iloc[-1]
    
    try_scored = int((group['actionName'] == 'Try').any())

    # Skip group if possession changed
    if first_row['teamName'] != last_row2['Possession']:
        return None  

    # Skip group if no movement occurred
    if first_row.equals(last_row):
        return None  

    # Skip if start position is in own half
    if first_row['x_coord'] <= 50:
        return None  
    
    # Calculate num_carries, directness, path_area_root
    num_carries = (group['actionName'] == 'Carry').sum()
    # Compute total distance (sum of segment distances)
    valid_moves = group[group['x_coord_end'] != 0]
    total_distance = np.sum(
        np.sqrt((valid_moves['x_coord_end'] - valid_moves['x_coord'])**2 +
                (valid_moves['y_coord_end'] - valid_moves['y_coord'])**2)
    )

    # Compute direct distance (first to last valid point)
    direct_distance = np.sqrt(
        (last_row['x_coord_end'] - first_row['x_coord'])**2 +
        (last_row['y_coord_end'] - first_row['y_coord'])**2
    )

    # Compute path area (Shoelace formula)
    path_area = 0
    if len(valid_moves) > 1:
        x_vals = valid_moves['x_coord'].tolist() + [last_row['x_coord_end']]
        y_vals = valid_moves['y_coord'].tolist() + [last_row['y_coord_end']]
        path_area = 0.5 * abs(
            sum(x_vals[i] * y_vals[i+1] - x_vals[i+1] * y_vals[i] for i in range(len(x_vals) - 1))
        )
    path_area_root = np.sqrt(path_area)
    
    # Compute directness
    directness = 0 if direct_distance == 0 else total_distance / direct_distance
    # Calculate xT
    xT = calculate_xT(
        first_row.x_coord, first_row.y_coord, last_row.x_coord_end, last_row.y_coord_end,
        num_carries, directness, path_area_root, model
    )
    
    return xT

players_9_FR = combined_df.loc[(combined_df['playerpositionID']==9) & (combined_df['teamName']=='France'), 'playerName'].unique().tolist()
players_10_FR = combined_df.loc[(combined_df['playerpositionID']==10) & (combined_df['teamName']=='France'), 'playerName'].unique().tolist()

# Now create filtered versions of grouped_dict based on start x_coord range
def player_xT_dict_filtered(players, grouped_dict, actions, model, min_x, max_x):
    xT_dict = {}
    for key, group in grouped_dict.items():
        if not ((group['playerName'].isin(players)) & (group['actionName'].isin(actions))).any():
            continue
        first_row = group.iloc[0]
        if not (min_x <= first_row['x_coord'] < max_x):
            continue
        for player_name in group['playerName'].unique():
            if player_name not in players:
                continue
            xT = player_xT(player_name, actions, group, model)
            if xT is not None:
                if player_name not in xT_dict:
                    xT_dict[player_name] = []
                xT_dict[player_name].append(xT)
    return xT_dict

# Calculate for each zone
xT_9_50_78_fr = player_xT_dict_filtered(players_9_FR, grouped_dict, ['Playmaker Options', 'P'], xT_model, 50, 78)
xT_9_78_100_fr = player_xT_dict_filtered(players_9_FR, grouped_dict, ['Playmaker Options', 'P'], xT_model, 78, 100)
xT_10_50_78_fr = player_xT_dict_filtered(players_10_FR, grouped_dict, ['Playmaker Options', 'P'], xT_model, 50, 78)
xT_10_78_100_fr = player_xT_dict_filtered(players_10_FR, grouped_dict, ['Playmaker Options', 'P'], xT_model, 78, 100)

# Convert to DataFrames
def build_stats(xT_dict, label):
    return pd.DataFrame([{
        'playerName': player,
        'avg_xT': np.mean(xT),
        'num_plays': len(xT),
        'zone': label
    } for player, xT in xT_dict.items()])

df_9_50_78_fr = build_stats(xT_9_50_78_fr, '9: 50-78')
df_9_78_100_fr = build_stats(xT_9_78_100_fr, '9: 78-100')
df_10_50_78_fr = build_stats(xT_10_50_78_fr, '10: 50-78')
df_10_78_100_fr = build_stats(xT_10_78_100_fr, '10: 78-100')

# Define selected French 9s
selected_9s = ['Antoine Dupont', 'Maxime Lucu', 'Baptiste Couilloud', 'Nolann Le Garrec']

# # Filter original DataFrames in-place and sort by avg_xT
df_9_50_78_fr = df_9_50_78_fr[df_9_50_78_fr['playerName'].isin(selected_9s)].sort_values(by='avg_xT', ascending=False)
df_9_78_100_fr = df_9_78_100_fr[df_9_78_100_fr['playerName'].isin(selected_9s)].sort_values(by='avg_xT', ascending=False)

# Plot 50–78
fig, ax = plt.subplots(figsize=(5, 5))
fig.set_facecolor('whitesmoke')

bars = ax.bar(df_9_50_78_fr['playerName'], df_9_50_78_fr['avg_xT'], color='navy')

for bar, num in zip(bars, df_9_50_78_fr['num_plays']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - 0.01,
            str(num), ha='center', va='bottom', fontproperties='serif', fontsize=11, color='white')

ax.set_xlabel('Player', fontproperties='serif')
ax.set_ylabel('Average xT', fontproperties='serif')
ax.set_title('Average xT in 50-78m (French Scrum-halves)', fontproperties='serif', fontsize=14)
ax.set_xticklabels(df_9_50_78_fr['playerName'], rotation=90, fontproperties='serif')
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot 78–100
fig, ax = plt.subplots(figsize=(5, 5))
fig.set_facecolor('whitesmoke')

bars = ax.bar(df_9_78_100_fr['playerName'], df_9_78_100_fr['avg_xT'], color='navy')

for bar, num in zip(bars, df_9_78_100_fr['num_plays']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - 0.025,
            str(num), ha='center', va='bottom', fontproperties='serif', fontsize=11, color='white')

ax.set_xlabel('Player', fontproperties='serif')
ax.set_ylabel('Average xT', fontproperties='serif')
ax.set_title('Average xT in the 22 (French Scrum-halves)', fontproperties='serif', fontsize=14)
ax.set_xticklabels(df_9_78_100_fr['playerName'], rotation=90, fontproperties='serif')
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

## PLOT OF SET OF PLAYS
chain = combined_df.loc[(combined_df.FXID==941369)&(combined_df.SetNum==42)]
chain = chain.loc[~(chain.actionName.isin(['Goal Kick', 'Restart']))&(chain.x_coord_end!=0)&(chain.PLID!=0)]
thetry = chain.iloc[-1]
start = chain.iloc[0]

# Step 1: Calculate xT for each PlayNum group
xT_values = {}

for play_num, group in chain.groupby('PlayNum'):
    first_row = group.iloc[0]
    last_row = group.iloc[-1]

    num_carries = (group['actionName'] == 'Carry').sum()

    total_distance = ((group['x_coord_end'] - group['x_coord'])**2 + (group['y_coord_end'] - group['y_coord'])**2).sum()**0.5
    direct_distance = ((last_row.x_coord_end - first_row.x_coord)**2 + (last_row.y_coord_end - first_row.y_coord)**2)**0.5
    directness = total_distance / direct_distance if direct_distance != 0 else 0

    path_x = group['x_coord'].tolist() + [group.iloc[-1]['x_coord_end']]
    path_y = group['y_coord'].tolist() + [group.iloc[-1]['y_coord_end']]

    # Approximate path area using Shoelace formula
    path_area = 0.5 * abs(sum(path_x[i] * path_y[i+1] - path_x[i+1] * path_y[i] for i in range(len(path_x) - 1)))
    path_area_root = path_area ** 0.5

    xT = calculate_xT(
        first_row.x_coord, first_row.y_coord,
        last_row.x_coord_end, last_row.y_coord_end,
        num_carries, directness, path_area_root, xT_model
    )

    xT_values[play_num] = xT  # Store xT value for this PlayNum


# Step 2: Assign colours based on PlayNum
unique_playnums = sorted(chain['PlayNum'].unique())
playnum_colours = {play_num: cm.plasma(i / len(unique_playnums)) for i, play_num in enumerate(unique_playnums)}

# Step 3: Plot using plt.Arrow
fig, ax = create_rotated_pitch(figsize=(7, 3), lwd=0.7)
fig.set_facecolor('whitesmoke')  # Background

# Plot each action in the chain
for _, row in chain.iterrows():
    play_xT = xT_values[row.PlayNum]
    colour = playnum_colours[row.PlayNum]

    if row.actionName == 'Pass':
        arrow = plt.Arrow(row.y_coord, row.x_coord,
                          row.y_coord_end - row.y_coord,
                          row.x_coord_end - row.x_coord,
                          width=2, color=colour, zorder=3)
    else:  # Carry
        arrow = plt.Arrow(row.y_coord, row.x_coord,
                          row.y_coord_end - row.y_coord,
                          row.x_coord_end - row.x_coord,
                          width=2, color=colour, linestyle='dashed', zorder=3)

    ax.add_patch(arrow)

# Highlight try
ax.scatter(thetry.y_coord_end, thetry.x_coord_end - 0.5, color="crimson", marker='x', s=100, linewidth=2, zorder=10)
ax.text(thetry.y_coord_end + 1, thetry.x_coord_end - 1, 'Try',
        color='crimson', zorder=10, size='large', weight='semibold', fontproperties='serif')

# Highlight start of play
ax.scatter(start.y_coord, start.x_coord, color='navy', marker='o', s=100, zorder=10)
ax.text(start.y_coord - 8, start.x_coord - 0.5, 'Scrum',
        color='navy', zorder=10, size='large', weight='semibold', fontproperties='serif')

# Create legend
legend_patches = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=playnum_colours[pn],
            markersize=8, label=f'{pn}: xT {xT_values[pn]:.3f}')
    for pn in unique_playnums
]

ax.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0.05, 0.2),
          frameon=False, prop={'family': 'serif', 'size': 8})

# Title and limits
ax.set_title('The xT of Different Movements Within a Set of Play', fontproperties='serif', fontsize=14)
ax.set_ylim(75, 101)
plt.show()


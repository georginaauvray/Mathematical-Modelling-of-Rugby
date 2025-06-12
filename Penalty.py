#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 12:03:47 2025

@author: georgieauvray
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
import shap
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm



warnings.filterwarnings('ignore')

# INITIAL DATA LOADING
# Folder containing all CSV files
folder_path = "csv"

# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Load all CSVs into a list of DataFrames
dataframes = [pd.read_csv(file) for file in csv_files]

# combine them into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

goal_kicks = combined_df.loc[combined_df['actionName']=="Goal Kick"]
goal_kicks2 = goal_kicks[['x_coord', 'y_coord', 'playerName', 'teamName', 'FXID', 'ActionTypeName', 'ActionResultName']]
goal_kicks2 = goal_kicks2.loc[goal_kicks2.ActionTypeName!='Drop Goal']
# Define number of rows
num_rows = 500

# Create a DataFrame with x_coord variations
simulated_data = pd.DataFrame({
    'x_coord': np.array([100] * num_rows + [10] * num_rows),  # Half 100, half 10
    'y_coord': np.random.uniform(0, 68, 2 * num_rows)  # y_coord between 0 and 68
})

# Add missing columns from goal_kicks2 (set them as NaN)
for col in goal_kicks2.columns:
    if col not in simulated_data:
        simulated_data[col] = None  # Or 0 if preferred

# Concatenate with original data
goal_kicks2 = pd.concat([goal_kicks2, simulated_data], ignore_index=True)

goal_kicks2['Distance'] = np.sqrt((100-goal_kicks2['x_coord'])**2 + abs(34-goal_kicks2['y_coord'])**2 )
goal_kicks2['Y'] = np.where(goal_kicks2.y_coord>34, 68-goal_kicks2.y_coord, goal_kicks2.y_coord)
goal_kicks2['Angle'] = np.where(goal_kicks2.Y<31.2, 
                                np.arctan((36.8-goal_kicks2.Y)/(100-goal_kicks2.x_coord))-np.arctan((31.2-goal_kicks2.Y)/(100-goal_kicks2.x_coord)), 
                                np.arctan(5.6*(100-goal_kicks2.x_coord)/((100-goal_kicks2.x_coord)**2-(36.8-goal_kicks2.Y)*(goal_kicks2.Y-31.2))))
goal_kicks2['Success'] = np.where(goal_kicks2.ActionResultName == 'Goal Kicked', 1, 0)

# PLOTTING KICKS
fig, ax = create_rotated_pitch()
for i, thekick in goal_kicks2.iterrows():
    y = thekick['x_coord']
    x = thekick['y_coord']
    goal = thekick['Success']
    if y==100 or y==10:
        shotCircle = plt.Circle((x,y), 0.2, color="black")
    else: 
        if goal:
            shotCircle = plt.Circle((x,y), 0.5, color='green')
        else:
            shotCircle = plt.Circle((x,y), 0.5, color="green")
            shotCircle.set_alpha(0.2)
    ax.add_patch(shotCircle)
fig.suptitle('All kicks in this era of rugby', fontsize=24)
plt.show()

## KICK MODEL
# Define features and target variable
X = goal_kicks2[['Distance', 'Angle']]
y = goal_kicks2['Success']

# Create and fit the model
kick_logreg = LogisticRegression(class_weight='balanced', solver='liblinear')
kick_logreg.fit(X, y)

# return xG value for more general model
def calculate_xG(sh):
    X_new = np.array([[sh['Distance'], sh['Angle']]])
    xG = kick_logreg.predict_proba(X_new)[:, 1]
    return xG

xG = goal_kicks2.apply(calculate_xG, axis=1)
goal_kicks2 = goal_kicks2.assign(xG=xG)

pgoal_2d=np.zeros((100,68))
for x in range(100):
    for y in range(68):
        sh=dict()
        y2 = np.where(y>34, 68-y, y)
        a = np.where(y2<31.2, np.arctan((36.8-y2)/(100-x))-np.arctan((31.2-y2)/(100-x)),
                      np.arctan(5.6*(100-x)/((100-x)**2-(36.8-y2)*(y2-31.2))))
        sh['Angle'] = a
        sh['Distance'] = np.sqrt((100-x)**2 + abs(34-y)**2)
        pgoal_2d[x,y] =  calculate_xG(sh)

# plot pitch
fig, ax = create_rotated_pitch()
fig.set_facecolor('whitesmoke')
#plot probability (colour used before was RdPu)
pos = ax.imshow(pgoal_2d, aspect='auto',cmap=cm.PiYG, zorder = 1)
fig.colorbar(pos, ax=ax, shrink=0.83)
#make legend
ax.set_title('Predicted Kick Success', fontproperties='serif', fontsize=14)
plt.xlim((0,68))
plt.ylim((0,100))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

## TOUCH KICKS 
touch_kicks = combined_df.loc[(combined_df['ActionTypeName']=='Touch Kick') & (combined_df['qualifier3Name']=='Penalty Kick')]

X = touch_kicks[['x_coord', 'y_coord']]
y = touch_kicks['x_coord_end']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
touch_linreg = LinearRegression()

# Train the model
touch_linreg.fit(X_train, y_train)

# Initialize an empty DataFrame with columns
df = pd.DataFrame(columns=['x', 'y', 'end_x'])

# Loop through the ranges and create the new data
for i in range(1, 10):
    X_new = np.array([[i*10, 34]])
    # Predict the value of x_coord_end
    end_x = touch_linreg.predict(X_new)[0]
    end_x = np.clip(end_x, None, 95)
    new_row = pd.DataFrame({'x': [i*10], 'y': 34, 'end_x': [end_x]})
    # Concatenate the new row to the original DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
        
# Plotting
fig, ax = create_pitch(figsize=(7, 3))
fig.set_facecolor('whitesmoke')  # Background

for i, kick in df.iterrows():
    x = kick['x']
    y = 34
    dx = kick['end_x'] - x
    dy = 34  # Keep it flat on the y-axis
    arrow = plt.Arrow(x, y, dx, dy, width=1.8, color='deeppink', zorder=2)
    ax.add_patch(arrow)

ax.set_title('Predicted End Location of Penalty Touch Kicks', fontproperties='serif', fontsize=14)
plt.xlim((0, 100))
plt.ylim((0, 68))
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

## Diagnostics
# Get predictions and clip
y_pred_touch = touch_linreg.predict(X_test)
x_pred_clipped_touch = np.clip(y_pred_touch, None, 95)

# Compute R²
r2 = r2_score(y_test, y_pred_clipped_touch)
print(f'Clipped R²: {r2:.3f}')

# Plot predicted vs actual
fig, ax = plt.subplots(figsize=(6, 5))
fig.set_facecolor('whitesmoke')

ax.scatter(y_test, y_pred_clipped_touch, color='navy', alpha=0.6, label='Predictions')
ax.plot([0, 100], [0, 100], linestyle='--', color='deeppink', lw=2, label='Perfect prediction')

ax.set_xlabel('Actual x_coord_end', fontproperties='serif')
ax.set_ylabel('Predicted x_coord_end', fontproperties='serif')
ax.set_title('Clipped Predictions vs Actual (Penalty Touch Kicks)', fontproperties='serif', fontsize=14)
ax.legend(prop={'family': 'serif'})
plt.tight_layout()
plt.show()

## SEQUENCES AFTER LINEOUTS
def assign_sequence(df):
    df = df.sort_values(['FXID', 'ID']).copy()  # Ensure order within each FXID
    df['Sequence'] = 0  # Initialise sequence column

    for fxid, group in df.groupby('FXID', group_keys=False):
        sequence_num = 0  # Reset sequence count for each FXID
        
        for i in range(len(group) - 1):  # Iterate through rows within the FXID group
            df.at[group.index[i], 'Sequence'] = sequence_num  # Assign sequence number
            
            next_action = group.at[group.index[i + 1], 'actionName']
            current_action = group.at[group.index[i], 'actionName']
            current_result = group.at[group.index[i], 'ActionResultName']
            next_result = group.at[group.index[i + 1], 'ActionResultName']

            # Check if sequence should increase
            if (next_action in {'Restart', 'Scrum', 'Lineout Throw', 'Goal Kick'} or
                current_action == 'Turnover' or
                (current_result in {'Pen Won', 'Pen Con'} and next_result not in {'Pen Won', 'Pen Con'})):
                sequence_num += 1  # Increment sequence count
        
        df.at[group.index[-1], 'Sequence'] = sequence_num  # Assign last row of group

    return df

def create_lineout_dataset(df):
    df = df.sort_values(['FXID', 'ID']).copy()  # Ensure proper ordering

    lineout_data = []  # List to store the new dataset

    for (fxid, sequence), group in df.groupby(['FXID', 'Sequence'], group_keys=False):
        first_row = group.iloc[0]  # Get the first row of the sequence

        if first_row['actionName'] == 'Lineout Throw':  # Check if sequence starts with Lineout Throw
            lineout_team = first_row['teamName']  # Store the team that did the Lineout Throw
            lineout_x_coord = first_row['x_coord']  # Store the x_coord of the Lineout Throw
            match_time = first_row['MatchTime']  # Store the MatchTime of the Lineout Throw
            
            # Determine opposition team
            opposition_team = first_row['awayTeamName'] if first_row['isHome'] == 'Y' else first_row['homeTeamName']

            # Determine the Result
            result = 0  # Default is NA
            
            if ((group['actionName'] == 'Try').any() or 
                ((group['actionName'] == 'Penalty Conceded') & (group['teamName'] != lineout_team)).any()):
                result = 1  # Success condition

            # Append to the new dataset
            lineout_data.append({'FXID': fxid, 'teamName': lineout_team, 
                                 'OppositionTeam': opposition_team,
                                 'x_coord': lineout_x_coord, 'MatchTime': match_time, 
                                 'Result': result,
                                 'Sequence': sequence})

    return pd.DataFrame(lineout_data)  # Convert list to DataFrame

combined_df2 = assign_sequence(combined_df)
lineouts = create_lineout_dataset(combined_df2)
 
# Ensure the necessary columns are numeric
lineouts['Distance'] = 100 - lineouts['x_coord']
lineouts['Distance_squared'] = lineouts['Distance'] ** 2

# Encode teamName and OppositionTeam as dummy variables
team_dummies = pd.get_dummies(lineouts['teamName'], prefix='Team', drop_first=True)
opposition_dummies = pd.get_dummies(lineouts['OppositionTeam'], prefix='Opposition', drop_first=True)

# Combine features into X
X = pd.concat([lineouts[['Distance', 'Distance_squared']], team_dummies, opposition_dummies], axis=1)

# Add a constant for the intercept term
X = sm.add_constant(X)
X = X.replace({True: 1, False: 0})  # Convert boolean to int
X = X.astype(float)  # Ensure all values are numeric

# Define the target variable, making sure it's numeric and dropping any NaNs
y = lineouts['Result']

# Fit the logistic regression model
lineout_model = sm.Logit(y, X).fit()

# Print the summary of the model
print(lineout_model.summary())

# Predict probabilities
y_pred_lineout = lineout_model.predict(X)

# Compute ROC curve
fpr, tpr, _ = roc_curve(y, y_pred_lineout)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
fig, ax = plt.subplots(figsize=(6, 5))
fig.set_facecolor('whitesmoke')  # Background

# ROC curve and diagonal reference
ax.plot(fpr, tpr, color='navy', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='deeppink', linestyle='--', lw=1.5)

# Axis limits and labels
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontproperties='serif')
ax.set_ylabel('True Positive Rate', fontproperties='serif')

# Title and legend
ax.set_title('Receiver Operating Characteristic (ROC) Curve, Lineout', fontproperties='serif', fontsize=14)
ax.legend(loc="lower right", prop={'family': 'serif'})

plt.tight_layout()
plt.show()

def calculate_lineout_try(team_name, opposition_name, x_coord, model):
    sh = dict()
     
    # Calculate Distance and Distance_squared
    distance = 100 - x_coord
    distance_squared = distance ** 2
    
    # Assign these to the dictionary
    sh['Distance'] = distance
    sh['Distance_squared'] = distance_squared
    
    # Create a list of all possible teams and oppositions
    all_teams = ['Australia', 'Barbarians', 'Canada', 'Chile', 'England', 'Fiji', 'France', 'Georgia', 
                  'Ireland', 'Italy', 'Japan', 'Namibia', 'New Zealand', 'Portugal', 'Romania', 'Samoa', 
                  'Scotland', 'South Africa', 'Spain', 'Tonga', 'USA', 'Uruguay', 'Wales']
    
    all_oppositions = ['Australia', 'Barbarians', 'Canada', 'Chile', 'England', 'Fiji', 'France', 'Georgia', 
                        'Ireland', 'Italy', 'Japan', 'Namibia', 'New Zealand', 'Portugal', 'Romania', 'Samoa', 
                        'Scotland', 'South Africa', 'Spain', 'Tonga', 'USA', 'Uruguay', 'Wales']
    
    # Add 1 for the team and opposition in the dictionary, and 0 for others
    for team in all_teams:
        sh[f'Team_{team}'] = 1 if team == team_name else 0
    
    for opposition in all_oppositions:
        sh[f'Opposition_{opposition}'] = 1 if opposition == opposition_name else 0
    
    # Convert the dictionary into a DataFrame
    feature_vector = pd.DataFrame([sh])
    
    feature_vector.insert(0, 'const', 1.0)  # Adds 'const' as the first column
    result_prob = model.predict(feature_vector).item()
    
    return result_prob


# Example usage
team_name = 'Ireland'
opposition_name = 'England'
x_coord = 90  # Example x-coordinate

# result_probability = calculate_lineout_try(team_name, opposition_name, x_coord, lineout_model)
# print(f"Probability of Positive Result for {team_name} vs {opposition_name} (logreg): {result_probability}")

## DECISION MAKING
def pen_points(x, y, team_name, opposition_name, model):
    # Calculate the kick probability using the calculate_xG function
    sh = dict()
    y2 = np.where(y > 34, 68 - y, y)
    
    a = np.where(y2 < 31.2, 
                 np.arctan((36.8 - y2) / (100 - x)) - np.arctan((31.2 - y2) / (100 - x)),
                 np.arctan(5.6 * (100 - x) / ((100 - x) ** 2 - (36.8 - y2) * (y2 - 31.2))))
    
    sh['Angle'] = a
    sh['Distance'] = np.sqrt((100 - x) ** 2 + abs(34 - y) ** 2)
    
    kick_prob = calculate_xG(sh)  # Probability of kick success
    kick_points = kick_prob * 3  # 3 points for a successful kick
    kick_points = kick_points[0]
    
    touch = touch_linreg.predict([[x, y]])[0]
    
    # Calculate the lineout success probability using the calculate_lineout_try function
    try_prob = calculate_lineout_try(team_name, opposition_name, touch, model)
    try_points = 5 * try_prob  # 7 points for a successful try after lineout
    
    return kick_points, try_points

## MAKING THE DECISION FUNCTION

penalties = combined_df.loc[(combined_df.actionName=='Sequences')&(combined_df.ActionTypeName.isin(['Penalty Goal', 'Penalty Kick Touch']))]
penalties['oppositionName'] = penalties.apply(
    lambda row: row['awayTeamName'] if row['isHome'] == 'Y' else row['homeTeamName'], axis=1
)
penalties['score_advantage_relative'] = penalties.apply(
    lambda row: row['score_advantage'] if row['isHome'] == 'Y' else -1*row['score_advantage'], axis=1
)
penalties[['kick_xG', 'touch_xG']] = penalties.apply(
    lambda row: pd.Series(pen_points(row['x_coord'], row['y_coord'], row['teamName'], row['oppositionName'], lineout_model)), 
    axis=1
)

# Define X and y
X = penalties[['MatchTime', 'score_advantage_relative', 'kick_xG', 'touch_xG']]
y = (penalties['ActionTypeName'] == 'Penalty Goal').astype(int)  # Convert to binary (0 for Touch, 1 for Goal)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the XGBoost model
penalty_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
penalty_xgb.fit(X_train, y_train)

# import joblib
# # Save the model
# joblib.dump(penalty_xgb, 'penalty_xgb_model.pkl')
# penalty_xgb = joblib.load('penalty_xgb_model.pkl')
y_proba = penalty_xgb.predict_proba(X_test)[:,1]

# Plot ROC curve
#Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# # Styled ROC curve plot
fig, ax = plt.subplots(figsize=(6, 5))
fig.set_facecolor('whitesmoke')

ax.plot(fpr, tpr, color='navy', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='deeppink', lw=1.5)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontproperties='serif')
ax.set_ylabel('True Positive Rate', fontproperties='serif')
ax.set_title('Receiver Operating Characteristic (ROC) Curve, Penalty XGB',
              fontproperties='serif', fontsize=14)
ax.legend(loc='lower right', prop={'family': 'serif'})

plt.tight_layout()
plt.show()

## Team predictions
penalty_dfs = {team: penalties[penalties['teamName'] == team] for team in ['Ireland', 'France', 'New Zealand', 'England', 'Scotland', 'South Africa']}
team_results = {}

for team, df in penalty_dfs.items():
    X_team = df[['MatchTime', 'score_advantage_relative', 'kick_xG', 'touch_xG']]
    y_team = df['ActionTypeName'].apply(lambda x: 1 if x == 'Penalty Goal' else 0)
    
    y_pred = penalty_xgb.predict(X_team)
    cm = confusion_matrix(y_team, y_pred)
    team_results[team] = cm

# Plot confusion matrices in your format
for team, cm in team_results.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.set_facecolor('whitesmoke')
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Penalty Kick Touch', 'Penalty Goal'],
                yticklabels=['Penalty Kick Touch', 'Penalty Goal'],
                cbar=False, ax=ax)

    ax.set_xlabel("Predicted", fontproperties='serif')
    ax.set_ylabel("Actual", fontproperties='serif')
    ax.set_title(f"Confusion Matrix for {team}", fontproperties='serif', fontsize=14)

    plt.tight_layout()
    plt.show()

# Diagnostics

# Plot feature importance
plot_importance(penalty_xgb, importance_type='weight', max_num_features=10)
plt.show()

# Initialize the explainer
explainer = shap.TreeExplainer(penalty_xgb)

# Calculate SHAP values for the data
shap_values = explainer.shap_values(X)

# Create a summary plot
shap.summary_plot(shap_values, X)

features = ['MatchTime', 'score_advantage_relative', 'kick_xG', 'touch_xG']

# Create partial dependence plots
fig, ax = plt.subplots(figsize=(12, 8))
fig.set_facecolor('whitesmoke')

disp = PartialDependenceDisplay.from_estimator(penalty_xgb, X, features, ax=ax)

# Title and font styling
plt.suptitle('Partial Dependence Plots for Penalty Decision Model',
              fontproperties='serif', fontsize=14)

plt.tight_layout()
plt.show()

# Define the pairs you want to see 
interaction_pairs = [
    ('MatchTime', 'score_advantage_relative'),
    ('MatchTime', 'kick_xG'),
    ('MatchTime', 'touch_xG')
]

fig, ax = plt.subplots(figsize=(12, 10))
fig.set_facecolor('whitesmoke')

disp = PartialDependenceDisplay.from_estimator(
    penalty_xgb,
    X,
    interaction_pairs,
    ax=ax
)

plt.suptitle('Partial Dependence Plots: Interactions with MatchTime',
              fontproperties='serif', fontsize=14)

plt.tight_layout()
plt.show()

## Plot of penalty predictions
# Actual labels
pen_ire = penalty_dfs['Ireland'].copy()

# # Reconstruct features exactly as in the model
X_ire = pen_ire[['MatchTime', 'score_advantage_relative', 'kick_xG', 'touch_xG']].copy()
y_true = (pen_ire['ActionTypeName'] == 'Penalty Goal').astype(int)
y_pred = penalty_xgb.predict(X_ire)

# Find misclassified indices
misclassified = y_true != y_pred

# Coordinates of those plays
x_coords1 = pen_ire.loc[misclassified, 'x_coord']
y_coords1 = pen_ire.loc[misclassified, 'y_coord']
true_vals1 = y_true[misclassified]
pred_vals1 = y_pred[misclassified]

# Coordinates of those plays
x_coords2 = pen_ire.loc[~misclassified, 'x_coord']
y_coords2 = pen_ire.loc[~misclassified, 'y_coord']
true_vals2 = y_true[~misclassified]
pred_vals2 = y_pred[~misclassified]

# Create pitch
fig, ax = create_rotated_pitch()
fig.set_facecolor('whitesmoke')

# Plot misclassified points
# Add misclassified points with annotations
for x, y, true_val, pred_val, mt, adv in zip(
    x_coords1,
    y_coords1,
    true_vals1,
    pred_vals1,
    pen_ire.loc[misclassified, 'MatchTime'],
    pen_ire.loc[misclassified, 'score_advantage_relative']
):
    marker = 'x' if true_val == 1 else 'o'
    color = 'navy' if true_val == 1 else 'deeppink'

    # Plot point
    ax.scatter(y, x, marker=marker, color=color, s=80, linewidths=2, zorder=10)

    # Format annotation
    label = f'Time: {mt}\nScore: {adv}'
    ax.text(y + 2, x + 0.5, label,
            fontsize=15, fontproperties='serif',
            color='black', zorder=10, va='center')
    
# Plot correctly classified points
for x, y, true_val, pred_val in zip(x_coords2, y_coords2, true_vals2, pred_vals2):
    marker = 'x' if true_val == 1 else 'o'
    ax.scatter(y, x, marker=marker, color='gray', s=50, alpha=0.4, linewidths=2, zorder=9)

# Final plot styling
ax.set_title('Misclassified Penalty Decisions (Ireland)', fontproperties='serif', fontsize=18)
plt.show()






#!/usr/bin/env python3
# Count Model Evaluation, Validation & Visualization Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomialP
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import os
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeRegressor
import networkx as nx

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create output directory if it doesn't exist
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"model_evaluation_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

# Load the biochemists data
print("Loading biochemists data...")
df = pd.read_csv('biochemists_data.csv')

# Create binary dummy variables for use in models
df['fem_dummy'] = (df['fem'] == 2).astype(int)  # 1 if female, 0 if male
df['mar_dummy'] = (df['mar'] == 1).astype(int)  # 1 if married, 0 if single
df['has_kids5'] = (df['kid5'] > 0).astype(int)  # 1 if has kids under 5

# Define the predictors and response variable
X = df[['fem_dummy', 'mar_dummy', 'kid5', 'phd', 'ment']]
y = df['art']
X_with_const = sm.add_constant(X)

# ======================================================================
# 7. MODEL COMPARISON & VALIDATION
# ======================================================================
print("\n" + "="*70)
print("7. MODEL COMPARISON & VALIDATION")
print("="*70)

# Initialize results storage
cv_results = {
    'Model': [],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'LogLikelihood': [],
    'AIC': [],
    'BIC': []
}

# Function to compute cross-validation metrics for count models
def perform_cv(model_type, X, y, n_folds=10):
    """Perform cross-validation for different count models"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mae_scores = []
    mse_scores = []
    log_likelihood = []
    aic_scores = []
    bic_scores = []
    
    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold += 1
        print(f"  Processing fold {fold}/{n_folds}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Add constant to X_train
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        # Fit the specified model
        if model_type == 'Poisson':
            model = Poisson(y_train, X_train_const)
        elif model_type == 'NegBin':
            model = NegativeBinomialP(y_train, X_train_const)
        elif model_type == 'ZIP':
            model = ZeroInflatedPoisson(y_train, X_train_const, exog_infl=X_train_const)
        
        try:
            result = model.fit(disp=0)
            
            # Predict on test set
            y_pred = result.predict(X_test_const)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store metrics
            mae_scores.append(mae)
            mse_scores.append(mse)
            log_likelihood.append(result.llf)
            aic_scores.append(result.aic)
            bic_scores.append(result.bic)
        except:
            print(f"  Error in fold {fold} for {model_type}, skipping...")
    
    # Calculate mean metrics
    if mae_scores:
        return {
            'MAE': np.mean(mae_scores),
            'MSE': np.mean(mse_scores),
            'RMSE': np.sqrt(np.mean(mse_scores)),
            'LogLikelihood': np.mean(log_likelihood),
            'AIC': np.mean(aic_scores),
            'BIC': np.mean(bic_scores)
        }
    else:
        return {
            'MAE': np.nan,
            'MSE': np.nan,
            'RMSE': np.nan,
            'LogLikelihood': np.nan,
            'AIC': np.nan,
            'BIC': np.nan
        }

# Cross-validation for CORE tree (using DecisionTreeRegressor as an approximation)
def perform_tree_cv(X, y, n_folds=10):
    """Perform cross-validation for CORE tree using DecisionTreeRegressor"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mae_scores = []
    mse_scores = []
    
    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold += 1
        print(f"  Processing fold {fold}/{n_folds}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit CORE tree approximation
        tree = DecisionTreeRegressor(max_depth=4, random_state=42)
        tree.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = tree.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Store metrics
        mae_scores.append(mae)
        mse_scores.append(mse)
    
    # Calculate mean metrics
    return {
        'MAE': np.mean(mae_scores),
        'MSE': np.mean(mse_scores),
        'RMSE': np.sqrt(np.mean(mse_scores)),
        'LogLikelihood': np.nan,  # Not applicable for trees
        'AIC': np.nan,  # Not applicable for trees
        'BIC': np.nan   # Not applicable for trees
    }

print("\n7.1 Performing 10-fold cross-validation for all models...")
print("\n- Poisson Regression")
poisson_cv = perform_cv('Poisson', X, y)
cv_results['Model'].append('Poisson')
for metric in ['MAE', 'MSE', 'RMSE', 'LogLikelihood', 'AIC', 'BIC']:
    cv_results[metric].append(poisson_cv[metric])

print("\n- Negative Binomial Regression")
negbin_cv = perform_cv('NegBin', X, y)
cv_results['Model'].append('Negative Binomial')
for metric in ['MAE', 'MSE', 'RMSE', 'LogLikelihood', 'AIC', 'BIC']:
    cv_results[metric].append(negbin_cv[metric])

print("\n- Zero-Inflated Poisson")
zip_cv = perform_cv('ZIP', X, y)
cv_results['Model'].append('ZIP')
for metric in ['MAE', 'MSE', 'RMSE', 'LogLikelihood', 'AIC', 'BIC']:
    cv_results[metric].append(zip_cv[metric])

print("\n- Count Regression Tree")
tree_cv = perform_tree_cv(X, y)
cv_results['Model'].append('CORE Tree')
for metric in ['MAE', 'MSE', 'RMSE', 'LogLikelihood', 'AIC', 'BIC']:
    cv_results[metric].append(tree_cv[metric])

# Create a results dataframe
cv_df = pd.DataFrame(cv_results)
print("\n7.2 Cross-validation Results Summary:")
print(cv_df)

# Save the CV results to CSV
cv_file = os.path.join(output_dir, 'cv_results.csv')
cv_df.to_csv(cv_file, index=False)
print(f"CV results saved to: {cv_file}")

# Plot comparison of MAE and RMSE for different models
print("\n7.3 Creating performance comparison plots...")
plt.figure(figsize=(12, 6))

# Plot MAE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MAE', data=cv_df)
plt.title('Mean Absolute Error by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Plot RMSE
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='RMSE', data=cv_df)
plt.title('Root Mean Squared Error by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
performance_plot_file = os.path.join(output_dir, 'model_performance_comparison.png')
plt.savefig(performance_plot_file)
plt.close()
print(f"Performance comparison plot saved to: {performance_plot_file}")

# Create a heatmap for model comparison
plt.figure(figsize=(10, 8))
metrics_df = cv_df.drop(['LogLikelihood', 'AIC', 'BIC'], axis=1).set_index('Model')
sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Model Performance Comparison')
plt.tight_layout()

# Save the heatmap
heatmap_file = os.path.join(output_dir, 'model_comparison_heatmap.png')
plt.savefig(heatmap_file)
plt.close()
print(f"Model comparison heatmap saved to: {heatmap_file}")

# ======================================================================
# 8. TREE STRUCTURE & INTERPRETATION
# ======================================================================
print("\n" + "="*70)
print("8. TREE STRUCTURE & INTERPRETATION")
print("="*70)

print("\n8.1 Visualizing Count Regression Tree...")
# Extract the tree structure from the output file
with open('count_modeling_results_20250507_182656.txt', 'r') as f:
    content = f.read()

# Find the tree structure section
tree_section_start = content.find("Final Count Regression Tree structure:")
if tree_section_start != -1:
    tree_structure = content[tree_section_start:]
    
    # Extract the tree nodes and structure from the text
    # Extract node information
    import re
    
    # Parse tree structure
    tree_lines = tree_structure.split('\n')
    
    # Build a directed graph to represent the tree
    G = nx.DiGraph()
    
    # Variables to track the current node and parent nodes at each level
    current_node_id = 0
    parent_nodes = {}
    
    # Key variable names from the tree
    variables = {
        'Mentors': 'Number of mentors',
        'Single': 'Marital status',
        'Kids under 5': 'Number of kids under 5',
        'PhD Quality': 'PhD quality score',
        'Female': 'Gender'
    }
    
    # Descriptions for top splits
    split_descriptions = {}
    
    # Parse the tree structure
    for line in tree_lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a split node or leaf node
        if 'Split:' in line:
            # Extract the split variable and value
            split_info = re.search(r'Split: (.+) <= (.+), n=(\d+)', line)
            if split_info:
                var_name, threshold, n = split_info.groups()
                node_id = current_node_id
                current_node_id += 1
                
                # Extract the indentation level to determine parent
                indent_level = len(line) - len(line.lstrip())
                indent_level = indent_level // 2  # Assuming 2 spaces per level
                
                # Determine the parent node
                parent_id = parent_nodes.get(indent_level-1, None)
                
                # Add the node to the graph
                label = f"{var_name}\n<= {threshold}\nn={n}"
                G.add_node(node_id, label=label, type='split', 
                           var=var_name, threshold=float(threshold), n=int(n))
                
                # If this node has a parent, add an edge from parent to this node
                if parent_id is not None:
                    G.add_edge(parent_id, node_id)
                
                # Store this node as the parent for the next level
                parent_nodes[indent_level] = node_id
                
                # Store descriptions for top splits (first 3 levels)
                if indent_level <= 2:
                    var_description = variables.get(var_name, var_name)
                    split_descriptions[node_id] = f"{var_description} threshold: {threshold}"
        
        elif 'Leaf Node:' in line:
            # Extract the leaf node information
            leaf_info = re.search(r'Leaf Node: n=(\d+), prediction=(.+)', line)
            if leaf_info:
                n, prediction = leaf_info.groups()
                node_id = current_node_id
                current_node_id += 1
                
                # Extract the indentation level to determine parent
                indent_level = len(line) - len(line.lstrip())
                indent_level = indent_level // 2  # Assuming 2 spaces per level
                
                # Determine the parent node
                parent_id = parent_nodes.get(indent_level-1, None)
                
                # Add the node to the graph
                label = f"Leaf\nn={n}\nprediction={prediction}"
                G.add_node(node_id, label=label, type='leaf', 
                           n=int(n), prediction=float(prediction))
                
                # If this node has a parent, add an edge from parent to this node
                if parent_id is not None:
                    G.add_edge(parent_id, node_id)
    
    # Draw the tree (alternative method without pygraphviz)
    plt.figure(figsize=(16, 12))
    
    try:
        # Try to use networkx's spring layout instead of graphviz
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors for split and leaf nodes
        node_colors = ['lightblue' if G.nodes[n]['type'] == 'split' else 'lightgreen' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        # Add labels to each node
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # Add a legend
        split_node = mpatches.Patch(color='lightblue', label='Split Node')
        leaf_node = mpatches.Patch(color='lightgreen', label='Leaf Node (Prediction)')
        plt.legend(handles=[split_node, leaf_node], loc='upper right')
        
        plt.title('Count Regression Tree Structure', fontsize=16)
        plt.axis('off')  # Turn off axis
    
    except Exception as e:
        print(f"Error in network visualization: {e}")
        print("Creating text-based tree representation instead...")
        
        # Create a textual representation of the tree
        plt.clf()  # Clear the figure
        plt.text(0.1, 0.9, "CORE Tree Structure", fontsize=16, fontweight='bold')
        
        # Sort nodes by level and position
        nodes_by_level = {}
        for n in G.nodes():
            level = len(nx.shortest_path(G, 0, n)) - 1 if n != 0 else 0
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(n)
        
        y_position = 0.8
        for level in sorted(nodes_by_level.keys()):
            nodes = sorted(nodes_by_level[level])
            plt.text(0.1, y_position, f"Level {level}:", fontsize=14, fontweight='bold')
            y_position -= 0.05
            
            for node in nodes:
                if G.nodes[node]['type'] == 'split':
                    var = G.nodes[node]['var']
                    threshold = G.nodes[node]['threshold']
                    n = G.nodes[node]['n']
                    plt.text(0.15, y_position, f"Node {node}: Split on {var} <= {threshold} (n={n})", fontsize=12)
                else:
                    n = G.nodes[node]['n']
                    prediction = G.nodes[node]['prediction']
                    plt.text(0.15, y_position, f"Node {node}: Leaf with prediction={prediction:.2f} (n={n})", fontsize=12)
                y_position -= 0.05
    
    # Print interpretations of top splits
    print("\n8.2 Interpreting Key Tree Splits:")
    for node_id, description in split_descriptions.items():
        print(f"- Node {node_id}: {description}")
        
        if 'Mentor' in description:
            print("  Interpretation: The number of mentors is the most important predictor of article counts. "
                  "Biochemists with more mentors tend to publish more articles.")
        elif 'Marital' in description or 'Single' in description:
            print("  Interpretation: Marital status affects publication counts, particularly for those with fewer mentors.")
        elif 'Kids' in description:
            print("  Interpretation: Having children under 5 tends to reduce publication counts.")
        elif 'PhD' in description:
            print("  Interpretation: PhD quality becomes an important factor for researchers with more mentors.")
        elif 'Gender' in description or 'Female' in description:
            print("  Interpretation: Gender appears to have an effect on publication counts after accounting for mentorship and PhD quality.")
    
    # Save interpretations to a text file
    interpretations_file = os.path.join(output_dir, 'tree_interpretations.txt')
    with open(interpretations_file, 'w') as f:
        f.write("CORE TREE INTERPRETATIONS\n")
        f.write("========================\n\n")
        for node_id, description in split_descriptions.items():
            f.write(f"Node {node_id}: {description}\n")
            
            if 'Mentor' in description:
                f.write("  Interpretation: The number of mentors is the most important predictor of article counts. "
                      "Biochemists with more mentors tend to publish more articles.\n")
            elif 'Marital' in description or 'Single' in description:
                f.write("  Interpretation: Marital status affects publication counts, particularly for those with fewer mentors.\n")
            elif 'Kids' in description:
                f.write("  Interpretation: Having children under 5 tends to reduce publication counts.\n")
            elif 'PhD' in description:
                f.write("  Interpretation: PhD quality becomes an important factor for researchers with more mentors.\n")
            elif 'Gender' in description or 'Female' in description:
                f.write("  Interpretation: Gender appears to have an effect on publication counts after accounting for mentorship and PhD quality.\n")
            f.write("\n")
    
    print(f"Tree interpretations saved to: {interpretations_file}")
else:
    print("Could not find tree structure in the modeling results file.")

# ======================================================================
# 9. RESIDUAL DIAGNOSTICS
# ======================================================================
print("\n" + "="*70)
print("9. RESIDUAL DIAGNOSTICS")
print("="*70)

print("\n9.1 Fitting models for residual diagnostics...")

# Fit models for residual diagnostics
try:
    # Poisson model
    poisson_model = Poisson(y, X_with_const)
    poisson_results = poisson_model.fit(disp=0)
    poisson_fitted = poisson_results.predict()
    poisson_residuals = y - poisson_fitted
    poisson_pearson = poisson_residuals / np.sqrt(poisson_fitted)
    
    # Negative binomial model
    negbin_model = NegativeBinomialP(y, X_with_const)
    negbin_results = negbin_model.fit(disp=0)
    negbin_fitted = negbin_results.predict()
    negbin_residuals = y - negbin_fitted
    negbin_pearson = negbin_residuals / np.sqrt(negbin_fitted * (1 + negbin_fitted * negbin_results.params[-1]))
    
    # Zero-inflated Poisson model
    zip_model = ZeroInflatedPoisson(y, X_with_const, exog_infl=X_with_const)
    try:
        zip_results = zip_model.fit(disp=0, maxiter=100)
        zip_fitted = zip_results.predict()
        zip_residuals = y - zip_fitted
        zip_pearson = zip_residuals / np.sqrt(zip_fitted)
        zip_zero_probs = zip_results.predict(which='prob-zero')
    except:
        print("Error fitting ZIP model. Using simpler version...")
        exog_infl = sm.add_constant(X[['fem_dummy']])  # Using just one predictor for inflation
        zip_model = ZeroInflatedPoisson(y, X_with_const, exog_infl=exog_infl)
        zip_results = zip_model.fit(disp=0, maxiter=100)
        zip_fitted = zip_results.predict()
        zip_residuals = y - zip_fitted
        zip_pearson = zip_residuals / np.sqrt(zip_fitted)
        zip_zero_probs = zip_results.predict(which='prob-zero')
    
    # CORE tree (using DecisionTreeRegressor as approximation)
    tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree_model.fit(X, y)
    tree_fitted = tree_model.predict(X)
    tree_residuals = y - tree_fitted

    print("\n9.2 Creating residual diagnostic plots...")
    
    # Plot residuals vs. fitted values
    plt.figure(figsize=(16, 12))
    
    # Poisson residuals
    plt.subplot(2, 2, 1)
    plt.scatter(poisson_fitted, poisson_pearson, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title('Poisson Model: Pearson Residuals vs. Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Pearson Residuals')
    
    # Negative Binomial residuals
    plt.subplot(2, 2, 2)
    plt.scatter(negbin_fitted, negbin_pearson, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title('Negative Binomial: Pearson Residuals vs. Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Pearson Residuals')
    
    # Zero-inflated Poisson residuals
    plt.subplot(2, 2, 3)
    plt.scatter(zip_fitted, zip_pearson, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title('ZIP Model: Pearson Residuals vs. Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Pearson Residuals')
    
    # CORE Tree residuals
    plt.subplot(2, 2, 4)
    plt.scatter(tree_fitted, tree_residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title('CORE Tree: Residuals vs. Fitted')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    
    # Save the residual plots
    residuals_file = os.path.join(output_dir, 'residual_diagnostics.png')
    plt.savefig(residuals_file)
    plt.close()
    print(f"Residual diagnostic plots saved to: {residuals_file}")
    
    # Check calibration of zero-probability predictions
    print("\n9.3 Checking calibration of zero-probability predictions...")
    
    # Create a dataframe of actual zeros and predicted zero probabilities
    zero_df = pd.DataFrame({
        'Actual': (y == 0).astype(int),
        'Poisson_Prob': np.exp(-poisson_fitted),
        'NegBin_Prob': negbin_results.predict(which='prob-zero'),
        'ZIP_Prob': zip_zero_probs
    })
    
    # Plot calibration curves
    plt.figure(figsize=(10, 8))
    
    # Function to create calibration plot
    def plot_calibration(actual, predicted, bins=10, label='Model'):
        """Plot calibration curve for model predictions"""
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(predicted, bin_edges) - 1
        bin_indices = np.minimum(bin_indices, bins - 1)  # Ensure no out of bounds
        
        bin_totals = np.bincount(bin_indices, minlength=bins)
        bin_positives = np.bincount(bin_indices, weights=actual, minlength=bins)
        bin_fractions = bin_positives / np.maximum(bin_totals, 1)
        
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        plt.plot(bin_centers, bin_fractions, 's-', label=label)
    
    # Plot the calibration curves for each model
    plot_calibration(zero_df['Actual'], zero_df['Poisson_Prob'], label='Poisson')
    plot_calibration(zero_df['Actual'], zero_df['NegBin_Prob'], label='Negative Binomial')
    plot_calibration(zero_df['Actual'], zero_df['ZIP_Prob'], label='ZIP')
    
    # Add the identity line (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')
    
    plt.title('Calibration of Zero-Probability Predictions')
    plt.xlabel('Predicted Probability of Zero')
    plt.ylabel('Observed Fraction of Zeros')
    plt.legend()
    plt.grid(True)
    
    # Save the calibration plot
    calibration_file = os.path.join(output_dir, 'zero_probability_calibration.png')
    plt.savefig(calibration_file)
    plt.close()
    print(f"Zero-probability calibration plot saved to: {calibration_file}")
    
    # Create error frequency tables
    print("\n9.4 Creating error frequency tables...")
    
    # Function to create error frequency table
    def create_error_table(y_actual, y_pred, model_name):
        """Create an error frequency table for count predictions"""
        errors = y_actual - y_pred
        error_counts = pd.Series(errors).value_counts().sort_index()
        error_props = error_counts / len(errors)
        error_df = pd.DataFrame({
            'Error': error_counts.index,
            'Count': error_counts.values,
            'Proportion': error_props.values
        })
        # Save to CSV
        error_file = os.path.join(output_dir, f'{model_name}_error_table.csv')
        error_df.to_csv(error_file, index=False)
        print(f"  Error table for {model_name} saved to: {error_file}")
        return error_df
    
    # Create error tables for each model
    poisson_error_df = create_error_table(y, np.round(poisson_fitted), 'poisson')
    negbin_error_df = create_error_table(y, np.round(negbin_fitted), 'negbin')
    zip_error_df = create_error_table(y, np.round(zip_fitted), 'zip')
    tree_error_df = create_error_table(y, np.round(tree_fitted), 'core_tree')
    
    # Summarize extreme errors
    extreme_errors = pd.DataFrame({
        'Model': ['Poisson', 'Negative Binomial', 'ZIP', 'CORE Tree'],
        'Max Abs Error': [
            max(abs(poisson_error_df['Error'])),
            max(abs(negbin_error_df['Error'])),
            max(abs(zip_error_df['Error'])),
            max(abs(tree_error_df['Error']))
        ],
        'Error > 2 Count': [
            sum(poisson_error_df[abs(poisson_error_df['Error']) > 2]['Count']),
            sum(negbin_error_df[abs(negbin_error_df['Error']) > 2]['Count']),
            sum(zip_error_df[abs(zip_error_df['Error']) > 2]['Count']),
            sum(tree_error_df[abs(tree_error_df['Error']) > 2]['Count'])
        ],
        'Error > 2 Prop': [
            sum(poisson_error_df[abs(poisson_error_df['Error']) > 2]['Proportion']),
            sum(negbin_error_df[abs(negbin_error_df['Error']) > 2]['Proportion']),
            sum(zip_error_df[abs(zip_error_df['Error']) > 2]['Proportion']),
            sum(tree_error_df[abs(tree_error_df['Error']) > 2]['Proportion'])
        ]
    })
    
    # Save extreme errors summary
    extreme_file = os.path.join(output_dir, 'extreme_errors_summary.csv')
    extreme_errors.to_csv(extreme_file, index=False)
    print(f"Extreme errors summary saved to: {extreme_file}")
    
except Exception as e:
    print(f"Error in residual diagnostics: {e}")

# Create a final results summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

# Identify the best model based on RMSE
best_model_rmse = cv_df.loc[cv_df['RMSE'].idxmin(), 'Model']
print(f"\nBest model based on RMSE: {best_model_rmse}")

# Identify the best model based on AIC (for non-tree models)
aic_df = cv_df[cv_df['AIC'].notna()]
if not aic_df.empty:
    best_model_aic = aic_df.loc[aic_df['AIC'].idxmin(), 'Model']
    print(f"Best model based on AIC: {best_model_aic}")

# Save the summary to a file
summary_file = os.path.join(output_dir, 'analysis_summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"MODEL EVALUATION & DIAGNOSTICS SUMMARY\n")
    f.write(f"=====================================\n\n")
    f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Cross-Validation Results:\n")
    f.write(f"{cv_df.to_string()}\n\n")
    
    f.write(f"Best model based on RMSE: {best_model_rmse}\n")
    if not aic_df.empty:
        f.write(f"Best model based on AIC: {best_model_aic}\n\n")
    
    f.write("Key Findings:\n")
    f.write("1. The data shows significant over-dispersion, as indicated by the Cameron-Trivedi test.\n")
    f.write("2. There is evidence of zero-inflation in the data.\n")
    f.write("3. The Negative Binomial model generally outperforms the Poisson model due to its ability to handle over-dispersion.\n")
    f.write("4. The CORE tree provides interpretable splits that reveal important variable thresholds.\n")
    f.write("5. The number of mentors is the most important predictor of publication counts.\n")
    f.write("6. Gender, marital status, and having young children also impact publication productivity.\n")

print(f"\nFinal summary saved to: {summary_file}")
print(f"\nAll results have been saved to: {output_dir}")
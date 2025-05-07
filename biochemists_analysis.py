#!/usr/bin/env python3
# Biochemists Data Analysis Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Load the biochemists data
print("Loading biochemists data...")
df = pd.read_csv('biochemists_data.csv')

# 1. DATA OVERVIEW & CLEANING
print("\n" + "="*50)
print("1. DATA OVERVIEW & CLEANING")
print("="*50)

# Display basic info
print("\nDataFrame Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")

# Recode categorical variables as pandas categories with clear labels
print("\nRecoding categorical variables...")

# Recode 'fem' (gender)
df['fem'] = df['fem'].astype('category')
df['fem'] = df['fem'].cat.rename_categories({1: 'Male', 2: 'Female'})
print("\nGender Categories:")
print(df['fem'].value_counts())

# Recode 'mar' (marital status)
df['mar'] = df['mar'].astype('category')
df['mar'] = df['mar'].cat.rename_categories({1: 'Married', 2: 'Single'})
print("\nMarital Status Categories:")
print(df['mar'].value_counts())

# 2. DESCRIPTIVE STATISTICS
print("\n" + "="*50)
print("2. DESCRIPTIVE STATISTICS")
print("="*50)

# Compute and print mean, median, variance of art
print("\nArticle Count Statistics:")
art_mean = df['art'].mean()
art_median = df['art'].median()
art_variance = df['art'].var()

print(f"Mean: {art_mean:.2f}")
print(f"Median: {art_median:.2f}")
print(f"Variance: {art_variance:.2f}")

# Create and display a frequency table for zero vs. non-zero art
print("\nZero vs. Non-zero Article Counts:")
df['art_category'] = df['art'].apply(lambda x: 'Zero' if x == 0 else 'Non-zero')
art_freq = df['art_category'].value_counts()
art_prop = df['art_category'].value_counts(normalize=True)

freq_table = pd.DataFrame({
    'Count': art_freq,
    'Proportion': art_prop
})
print(freq_table)

# Tabulate and print group means of art by fem, mar, and binned phd
print("\nGroup Means of Articles:")
print("\nBy Gender:")
print(df.groupby('fem')['art'].mean())

print("\nBy Marital Status:")
print(df.groupby('mar')['art'].mean())

# Create binned phd
df['phd_bin'] = pd.cut(df['phd'], bins=[0, 2, 4, 6], labels=['Low', 'Medium', 'High'])
print("\nBy PhD Quality:")
print(df.groupby('phd_bin')['art'].mean())

# Create a cross-tabulation of gender and marital status
print("\nMean Articles by Gender and Marital Status:")
print(df.groupby(['fem', 'mar'])['art'].mean().unstack())

# 3. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*50)
print("3. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Plot 1: Histogram overlaid with density curve for art
print("\nCreating histogram with density curve...")
plt.figure(figsize=(8, 6))
sns.histplot(df['art'], kde=True)
plt.title('Distribution of Article Counts', fontsize=16)
plt.xlabel('Number of Articles', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('plot1_histogram_article_counts.png')
print("Plot 1 saved as 'plot1_histogram_article_counts.png'")
plt.close()

# Plot 2: Bar chart comparing zero vs. positive counts
print("\nCreating bar chart for zero vs. non-zero counts...")
plt.figure(figsize=(8, 6))
sns.countplot(x='art_category', data=df)
plt.title('Zero vs. Non-zero Article Counts', fontsize=16)
plt.xlabel('Article Count Category', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('plot2_bar_zero_nonzero.png')
print("Plot 2 saved as 'plot2_bar_zero_nonzero.png'")
plt.close()

# Plot 3: Boxplot of art by fem
print("\nCreating boxplot of articles by gender...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='fem', y='art', data=df)
plt.title('Article Counts by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('plot3_boxplot_gender.png')
print("Plot 3 saved as 'plot3_boxplot_gender.png'")
plt.close()

# Plot 4: Boxplot of art by mar
print("\nCreating boxplot of articles by marital status...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='mar', y='art', data=df)
plt.title('Article Counts by Marital Status', fontsize=16)
plt.xlabel('Marital Status', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('plot4_boxplot_marital.png')
print("Plot 4 saved as 'plot4_boxplot_marital.png'")
plt.close()

# Plot 5: Generate a scatter plot of phd vs. art
print("\nCreating scatter plot of PhD quality vs. article counts...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='phd', y='art', data=df, alpha=0.7)
plt.title('PhD Quality vs. Article Counts', fontsize=16)
plt.xlabel('PhD Quality', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('plot5_scatter_phd_art.png')
print("Plot 5 saved as 'plot5_scatter_phd_art.png'")
plt.close()

# Plot 6: Add a jittered version with regression line
print("\nCreating jittered scatter plot with regression line...")
plt.figure(figsize=(8, 6))
# Add jitter to avoid overplotting
x_jitter = df['phd'] + np.random.normal(0, 0.1, size=len(df))
y_jitter = df['art'] + np.random.normal(0, 0.1, size=len(df))
sns.scatterplot(x=x_jitter, y=y_jitter, alpha=0.7)
# Add a regression line
sns.regplot(x=x_jitter, y=y_jitter, scatter=False)
plt.title('PhD Quality vs. Article Counts (Jittered)', fontsize=16)
plt.xlabel('PhD Quality', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('plot6_scatter_jittered_phd_art.png')
print("Plot 6 saved as 'plot6_scatter_jittered_phd_art.png'")
plt.close()

# Plot 7: Boxplot of art by phd_bin
print("\nCreating boxplot of articles by PhD quality bins...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='phd_bin', y='art', data=df)
plt.title('Article Counts by PhD Quality Bins', fontsize=16)
plt.xlabel('PhD Quality Category', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('plot7_boxplot_phd_bins.png')
print("Plot 7 saved as 'plot7_boxplot_phd_bins.png'")
plt.close()

# Plot 8: Violin plot for mentor count vs. article count
print("\nCreating violin plot of articles by mentor count bins...")
plt.figure(figsize=(8, 6))
# Create binned mentor count
df['ment_bin'] = pd.cut(df['ment'], bins=[0, 5, 10, 20, 100], 
                     labels=['0-5', '6-10', '11-20', '20+'])
sns.violinplot(x='ment_bin', y='art', data=df)
plt.title('Article Counts by Mentor Count', fontsize=16)
plt.xlabel('Mentor Count', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('plot8_violin_mentor_counts.png')
print("Plot 8 saved as 'plot8_violin_mentor_counts.png'")
plt.close()

print("\nAnalysis complete. Eight individual plot files have been saved.")
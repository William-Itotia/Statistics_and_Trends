"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file, or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

def plot_relational_plot(df):
    """
    Plots line graph that looks at the average conversion rate
    vs total ads shown, for the different test groups ads and
    PSA with the total ads on the x column being log binned
    due to the uneven spread of data.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    bins = np.logspace(0, np.log10(df["total ads"].max()), num=10)
    df["log_ads_bin"] = pd.cut(
        df["total ads"],
        bins=bins,
        labels=[f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]
    )

    groups = df["test group"].unique()
    
    for i, group in enumerate(groups, 1):
        subset = df[df["test group"] == group]
        conversion_rates = subset.groupby("log_ads_bin")["converted"].mean() * 100

        plt.subplot(1, 2, i)
        sns.lineplot(x=conversion_rates.index, y=conversion_rates.values, marker="o")
        plt.xticks(rotation=45)
        plt.xlabel("Total Ads Shown (Log Binned)")
        plt.ylabel("Average Conversion Rate (%)")
        plt.title(f"Conversion Rate vs Ads (Test Group: {group})")
        plt.grid()
    
    plt.tight_layout()
    plt.savefig('relational_plot.png')

def plot_categorical_plot(df):
    """
    Plots bar plot that looks at conversion rate of total ads below and
    above 100 ads for both test groups PSA and ad.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    threshold = 100
    df["converted"] = df["converted"].astype(int)
    df["ads_category"] = df["total ads"].apply(
        lambda x: "Above 100 Ads" if x > threshold else "Below 100 Ads"
    )
    
    groups = ["ad", "psa"]
    for i, group in enumerate(groups):
        subset = df[df["test group"] == group]
        conversion_rates = subset.groupby("ads_category")["converted"].mean() * 100
        ax = plt.subplot(1, 2, i + 1)
        
        sns.barplot(x=conversion_rates.index, y=conversion_rates.values, palette="husl", ax=ax)

        for j, value in enumerate(conversion_rates.values):
            ax.text(j, value + 1, f"{value:.2f}%", ha="center", fontsize=12, color="black", fontweight="bold")
        
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title(f"Conversion Rate Above and Below {threshold} Ads\n(Test Group: {group})")
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('categorical_plot.png')

def plot_statistical_plot(df):
    """
    Plots correlation heatmaps for 'converted' and 'total ads' separately
    for test groups 'ad' and 'psa'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for ax, group in zip(axes, ["ad", "psa"]):
        subset = df[df["test group"] == group][["converted", "total ads"]]
        corr_matrix = subset.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title(f'Correlation Heatmap (Test Group: {group})')
    
    plt.tight_layout()
    plt.savefig('statistical_plot.png')

def statistical_analysis(df, col: str):
    """
    Computes statistical moments (mean, standard deviation, skewness,
    and excess kurtosis) for a given column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """
    Preprocessing function that does the following:
    - Displays first 5 rows
    - Shows summary statistics
    - Prints missing values count
    - Shows correlation matrix
    """
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    print(df[['total ads', 'most ads hour']].corr())
    
    return df

def writing(moments, col):
    """
    Prints the computed statistical moments for a given column.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    print('The data was right skewed and leptokurtic.')

def main():
    """
    Main function to execute the workflow: loading data,
    preprocessing, visualizations, and statistical analysis.
    """
    df = pd.read_csv('data.csv', index_col=0)
    df = preprocessing(df)
    col = 'total ads'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)

if __name__ == '__main__':
    main()

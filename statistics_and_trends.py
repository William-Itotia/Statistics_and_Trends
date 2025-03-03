"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns



def plot_relational_plot(df):
    """
    Plots line graph that looks at the average conversion rate vs total ads shown, 
    for the different test groups ads and psa with the total ads on the x column 
    being log binned due to the uneven spread of data. 
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
     # Define logarithmic bins based on total ads
    bins = np.logspace(0, np.log10(df["total ads"].max()), num=10)

    # Assign bins using pd.cut()
    df["log_ads_bin"] = pd.cut(df["total ads"], bins=bins, labels=[f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)])

    # Split into two groups (ad vs. psa)
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
    return




def plot_categorical_plot(df):
    """
    Plots bar plot that looks at conversion rate of total ads below and above 100
    ads for both test groups psa and ad
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    threshold = 100  # Define the threshold for ad count
    df["converted"] = df["converted"].astype(int)  # Convert boolean to int

    # Create categories
    df["ads_category"] = df["total ads"].apply(lambda x: "Above 100 Ads" if x > threshold else "Below 100 Ads")

    # Split by test group
    groups = ["ad", "psa"]

    for i, group in enumerate(groups):
        subset = df[df["test group"] == group]
        conversion_rates = subset.groupby("ads_category")["converted"].mean() * 100

        ax = plt.subplot(1, 2, i + 1)
        sns.barplot(x=conversion_rates.index, y=conversion_rates.values, palette="husl", ax=ax)

        # Add text labels on bars
        for j, value in enumerate(conversion_rates.values):
            ax.text(j, value + 1, f"{value:.2f}%", ha="center", fontsize=12, color="black", fontweight="bold")

        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title(f"Conversion Rate Above and Below {threshold} Ads\n(Test Group: {group})")
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    return




def plot_statistical_plot(df):
    """
    Plots correlation heatmaps for 'converted' and 'total ads' separately 
    for test groups 'ad' and 'psa'.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, group in zip(ax, ["ad", "psa"]):
        subset = df[df["test group"] == group][["converted", "total ads"]]
        corr_matrix = subset.corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

        ax.set_title(f'Correlation Heatmap (Test Group: {group})')

    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    return




def statistical_analysis(df, col: str):
    """
    Computes statistical moments (mean, standard deviation, skewness, excess kurtosis) for a given column.
    """
    mean = df['total ads'].mean()
    stddev = df['total ads'].std()
    skew = ss.skew(df['total ads'])
    excess_kurtosis = ss.kurtosis(df['total ads'])  # Excess kurtosis (subtracts 3 from Fisher kurtosis)

    return mean, stddev, skew, excess_kurtosis




def preprocessing(df):
    """
    Preporecessing function that does the following:
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
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Determine skewness and kurtosis category

    skew_label = "not skewed" if abs(moments[2]) < 0.5 else ("right skewed" if moments[2] > 0 else "left skewed")
    kurtosis_label = "mesokurtic" if abs(moments[3]) < 2 else ("leptokurtic" if moments[3] > 2 else "platykurtic")
    print('The data was right/left/not skewed and platy/meso/leptokurtic.')
    return


def main():
    df = pd.read_csv('data.csv', index_col=0)
    df = preprocessing(df)
    col = 'total ads'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)

    return


if __name__ == '__main__':
    main()

import pandas as pd
from statsmodels.graphics.correlation import plot_corr
import matplotlib.pyplot as plt

def print_players(df: pd.DataFrame) -> None:
    for year in range(2015, 2018 + 1):
        print(f"============== {year} =============")
        print("============== HIT ============== ")
        print(df.where(df['year'] == year).dropna().sort_values(by='H', ascending=False)[:10])
        print("============== AVG ============== ")
        print(df.where(df['year'] == year).dropna().sort_values(by='avg', ascending=False)[:10])
        print("============== HR ============== ")
        print(df.where(df['year'] == year).dropna().sort_values(by='HR', ascending=False)[:10])
        print("============== OBP ============== ")
        print(df.where(df['year'] == year).dropna().sort_values(by='OBP', ascending=False)[:10])
        print(f"============== {year} =============")


def print_highest_war_player_by_position(df: pd.DataFrame) -> None:
    df_2018 = df[df['year'] == 2018]
    print("============== The highest WAR player by position in 2018 ==============")
    print(df_2018.loc[df_2018.groupby(by='cp')['war'].idxmax()][['batter_name', 'war', 'cp', 'year']])
    print("============== The highest WAR player by position in 2018 ==============")


def correlation(df: pd.DataFrame) -> None:
    print("============== The highest correlation between salary ==============")
    df_numerical = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
    corr_matrix = df_numerical.corr()
    plot_corr(corr_matrix, xnames=df_numerical.columns)
    plt.show()
    print(f"The highest correlation between salary is {corr_matrix['salary'][corr_matrix.index != 'salary'].idxmax()}, {corr_matrix['salary'][corr_matrix.index != 'salary'].max()}")
    print("============== The highest correlation between salary ==============")


if __name__ == '__main__':
    df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
    print_players(df)
    print_highest_war_player_by_position(df)
    correlation(df)

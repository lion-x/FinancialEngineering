import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def symbol_to_path(symbol, base_dir="../data"):
    return os.path.join(base_dir, "{}.csv".format(symbol))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True,
                              usecols=['Date', 'Adj Close'], na_values=['nan'])

        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock Price", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns.iloc[0, :] = 0
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    return daily_returns


def test_run():
    start_date = '2009-01-01'
    end_date = '2012-12-31'
    dates = pd.date_range(start_date, end_date)
    print(dates)
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)

    daily_returns = compute_daily_returns(df)

    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(x=daily_returns['SPY'], y=daily_returns['XOM'], deg=1)
    plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, linestyle='-', color='r')
    print("beta_XOM: ", beta_XOM)
    print("alpha_XOM: ", alpha_XOM)
    plt.show()

    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    beta_GLD, alpha_GLD = np.polyfit(x=daily_returns['SPY'], y=daily_returns['GLD'], deg=1)
    plt.plot(daily_returns['SPY'], beta_GLD * daily_returns['SPY'] + alpha_GLD, linestyle='-', color='r')
    print("beta_GLD: ", beta_GLD)
    print("alpha_GLD: ", alpha_GLD)

    plt.show()

    print(daily_returns.corr(method='pearson'))


if __name__ == "__main__":
    test_run()

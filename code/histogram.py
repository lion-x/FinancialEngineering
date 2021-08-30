import pandas as pd
import os
import matplotlib.pyplot as plt


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
    start_date = '2010-01-01'
    end_date = '2020-12-31'
    dates = pd.date_range(start_date, end_date)
    print(dates)
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)

    plot_data(df['SPY'])

    daily_returns = compute_daily_returns(df)

    plot_data(daily_returns['SPY'], title="Daily returns", ylabel='Daily return')

    print(daily_returns)

    daily_returns_SPY_mean = daily_returns['SPY'].mean()

    print("mean = ", daily_returns_SPY_mean)

    daily_returns_SPY_std = daily_returns['SPY'].std()

    print("std = ", daily_returns_SPY_std)

    daily_returns['SPY'].hist(bins=40)
    plt.axvline(daily_returns_SPY_mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(daily_returns_SPY_mean + daily_returns_SPY_std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(daily_returns_SPY_mean - daily_returns_SPY_std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    daily_returns_SPY_kurtosis = daily_returns['SPY'].kurtosis()
    print("kurtosis = ", daily_returns_SPY_kurtosis)

    daily_returns['XOM'].hist(bins=40, label='XOM')
    daily_returns['SPY'].hist(bins=40, label='SPY')

    plt.show()


if __name__ == "__main__":
    test_run()

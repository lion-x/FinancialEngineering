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


def plot_selected(df, columns, start_index, end_index):
    df_selected = df.loc[start_index:end_index, columns]
    plot_data(df_selected, "Selected data")


def normalize_data(df):
    return df / df.iloc[0, :]


def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns.iloc[0, :] = 0
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    return daily_returns


def test_run():
    start_date = '2010-01-01'
    end_date = '2012-12-31'
    dates = pd.date_range(start_date, end_date)

    symbols = ['SPY', 'XOM', 'GOOG', 'GLD']

    df = get_data(symbols, dates)

    plot_data(df)

    print(df.mean())

    print(df.median())

    print(df.std())

    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')

    rm_SPY = pd.Series.rolling(df['SPY'], window=20).mean()

    rstd_SPY = pd.Series.rolling(df['SPY'], window=20).std()

    upper_band_SPY, lower_band_SPY = get_bollinger_bands(rm_SPY, rstd_SPY)

    rm_SPY.plot(label='Rolling mean', ax=ax)

    upper_band_SPY.plot(label='Upper band', ax=ax)
    lower_band_SPY.plot(label='lower band', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    dates = pd.date_range('2012-07-01', '2012-07-31')
    symbols = ["SPY", "XOM"]
    df = get_data(symbols, dates)
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")


if __name__ == "__main__":
    test_run()

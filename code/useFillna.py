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


def plot_data(df, title="Stock Price"):
    ax = df.plot(title=title, fontsize=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def test_run():
    start_date = '2012-01-01'
    end_date = '2012-12-31'
    dates = pd.date_range(start_date, end_date)
    print(dates)

    symbols = ['FAKE', 'FB']

    df = get_data(symbols, dates)

    print(df)

    plot_data(df)

    df.fillna(method='ffill', inplace=True)

    df.fillna(method='backfill', inplace=True)

    plot_data(df)


if __name__ == "__main__":
    test_run()

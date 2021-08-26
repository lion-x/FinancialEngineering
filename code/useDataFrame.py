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


def plot_selected(df, columns, start_index, end_index):
    df_selected = df.loc[start_index:end_index, columns]
    plot_data(df_selected, "Selected data")


def normalize_data(df):
    return df/df.iloc[0, :]


def test_run():
    start_date = '2019-01-01'
    end_date = '2020-12-31'
    dates = pd.date_range(start_date, end_date)
    print(dates)

    symbols = ['GOOG', 'IBM', 'GLD']

    df = get_data(symbols, dates)

    print(df.loc['2020-01-01':'2020-01-31'])

    print(df['GOOG'])
    print(df[['IBM', 'GLD']])

    print(df.loc['2020-01-01':'2020-01-31', ['SPY', 'IBM']])

    plot_data(df)

    plot_selected(df, ["SPY", "IBM"], "2019-01-01", "2019-12-31")

    print(df.iloc[0, :])
    df_norm = normalize_data(df)
    plot_data(df_norm)


if __name__ == "__main__":
    test_run()

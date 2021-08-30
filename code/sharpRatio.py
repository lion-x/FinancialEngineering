import pandas as pd
import os
import math


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


def normalize_data(df):
    return df / df.iloc[0, :]


def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns.iloc[0, :] = 0
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    return daily_returns


def get_risk_free_data():
    df = pd.read_csv('../data/RF.csv', index_col='Date', parse_dates=True,
                     usecols=['Date', 'RF'], na_values=['nan'])
    return df


def test_run():
    start_val = 1000000
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    dates = pd.date_range(start_date, end_date)
    symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
    df = get_data(symbols, dates)
    allocs = [0.4, 0.4, 0.1, 0.1]
    df = normalize_data(df)
    df = df * allocs
    df = df * start_val
    df['Port Value'] = df.sum(axis=1)
    df['Port Daily Return'] = compute_daily_returns(df['Port Value'])
    cum_ret = (df['Port Value'][-1] / df['Port Value'][0]) - 1
    print('Cum_ret: ', cum_ret)
    avg_daily_ret = df['Port Daily Return'].mean()
    std_daily_ret = df['Port Daily Return'].std()
    print('avg_daily_return: ', avg_daily_ret)
    print('std_daily_return: ', std_daily_ret)

    df_rf = get_risk_free_data()
    df = df.join(df_rf)

    print("Trading days: ", df['SPY'].count())
    sharpRatio = math.sqrt(252) * (df['Port Daily Return'] - df['RF']).mean() / std_daily_ret
    print('Sharp Ratio: ', sharpRatio)


if __name__ == "__main__":
    test_run()

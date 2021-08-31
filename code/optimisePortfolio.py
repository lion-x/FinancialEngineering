import pandas as pd
import numpy as np
import os
import math
import scipy.optimize as spo


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


def compute_sharp_ratio(allocs, df):
    df = normalize_data(df)
    df = df * allocs
    df['Port Value'] = df.sum(axis=1)
    df['Port Daily Return'] = compute_daily_returns(df['Port Value'])
    avg_daily_ret = df['Port Daily Return'].mean()
    std_daily_ret = df['Port Daily Return'].std()

    df_rf = get_risk_free_data()
    df = df.join(df_rf)
    trading_days = df['XOM'].count()
    sharpRatio = math.sqrt(trading_days) * (df['Port Daily Return'] - df['RF']).mean() / std_daily_ret

    return -sharpRatio


def cons(allocs):
    return np.sum(allocs) - 1


def optimiseSR(df, optimise_func):
    allocsguess = [0.25, 0.25, 0.25, 0.25]
    allocs_bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
    allocs_constraints = {'type': 'eq', 'fun': cons}
    min_result = spo.minimize(optimise_func, allocsguess, args=(df, ), method="SLSQP", options={'disp': True},
                              bounds=allocs_bounds, constraints=allocs_constraints)
    return min_result.x


def test_run():
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    dates = pd.date_range(start_date, end_date)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    df = get_data(symbols, dates)
    del df['SPY']

    allocs = optimiseSR(df, compute_sharp_ratio)
    print("Allocs: ")
    print("GOOG: ", allocs[0])
    print("AAPL: ", allocs[1])
    print("GLD: ", allocs[2])
    print("XOM: ", allocs[3])
    print("Sharp Ratio: ", -compute_sharp_ratio(allocs, df))


if __name__ == "__main__":
    test_run()

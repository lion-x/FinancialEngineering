from pandas_datareader import data
import datetime

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 12, 31)

spy = data.DataReader("SPY", "yahoo", start, end)

spy.to_csv("../data/SPY.csv")

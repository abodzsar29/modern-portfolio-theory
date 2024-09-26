import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import pypfopt.plotting as plotting
import cvxpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MPT:
    historical_data = None
    def __init__(self, tickers, startdate, enddate):
        MPT.historical_data = yf.download(tickers,
                              start="2023-1-1",
                              end="2023-12-31")
        MPT.historical_data = MPT.historical_data["Adj Close"]  # Filtering data for adjusted close prices pnly

    def get_mean_returns(self):
        # First uses .pct_change() to calculate daily percentage returns, then calculates average returns for the entire given
        # period using the Geometric Mean Return method
        mean_returns = expected_returns.mean_historical_return(MPT.historical_data)
        return mean_returns

    def get_covariance_matx(self):
        data_pct_diff = MPT.historical_data.pct_change()[1:]  # Calculates the percentage change between daily values
        cov_matrix = risk_models.sample_cov(MPT.historical_data)



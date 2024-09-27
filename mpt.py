import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import pypfopt.plotting as plotting
import cvxpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict


class MPT:
    def __init__(self, tickers: str, startdate: str, enddate: str):
        self.tickers: str = tickers
        self.startdate: str = startdate
        self.enddate: str = enddate
        self._historical_data: Optional['pd.DataFrame'] = None
        self._ef: Optional[EfficientFrontier] = None
        self._mean_returns: Optional['pd.Series'] = None
        self._cov_matrix: Optional['pd.DataFrame'] = None
        self._weights: Optional[Dict[str, float]] = None
        self.__ef_constrained: Optional[EfficientFrontier] = None  # For graphical representation

    def _get_adj_close_data(self) -> None:
        if self._historical_data is None:
            self._historical_data = yf.download(self.tickers, start=self.startdate, end=self.enddate)["Adj Close"]

    def _get_mean_returns(self) -> 'pd.Series':  # Calculate average annualized returns using Geometric Mean Return
        self._get_adj_close_data()
        self._mean_returns = expected_returns.mean_historical_return(self._historical_data, compounding=True)
        return self._mean_returns

    def _get_covariance_matx(self) -> 'pd.DataFrame':
        self.__cov_matrix = risk_models.sample_cov(self._historical_data)
        return self.__cov_matrix

    def _get_efficient_front(self) -> None:
        # Long positions only in portfolio
        self._ef = EfficientFrontier(self._get_mean_returns(), self._get_covariance_matx(), weight_bounds=(0, 1))

    def __get_max_sharpe_r(self) -> None:
        self._get_efficient_front()
        self._weights = self._ef.max_sharpe()  # Maximises sharp ratio

    def print_sharpe_info(self) -> None:
        self.__get_max_sharpe_r()
        print("Individual Stock Weightings in Portfolio:")
        [print(f"{stock}: {weight}") for stock, weight in self._weights.items()]  # Stock weighting in portfolio
        print("##################### \nPortfolio Performance Summarised:")
        print(self._ef.portfolio_performance(verbose=True))

    def display_results(self) -> None:
        self.__get_max_sharpe_r()  # Recomputing the self.ef variable in case of modified data
        self.__ef_constrained = EfficientFrontier(self._get_mean_returns(),  # Required to plot EF and data
                                                self._get_covariance_matx(),
                                                weight_bounds=(0, 1))
        self.__ef_constrained.add_constraint(lambda x: cvxpy.sum(x) == 1)  # Checking if investments are 100% of prtflio
        fig, ax = plt.subplots()
        plotting.plot_efficient_frontier(self.__ef_constrained, ax=ax, show_assets=True)
        ax.scatter(self._ef.portfolio_performance()[1],
                   self._ef.portfolio_performance()[0],
                   marker="*",
                   color="r",
                   s=200,
                   label="Tangency Portfolio")
        ax.legend()
        plt.show()


def main() -> None:
    tckers = "PEP ADBE LIN TMO MCD CSCO ACN GE IBM ABT VZ QCOM TXN PM CAT NOW WFC INTU NEE DHR"
    start = "2023-01-01"
    end = "2023-12-31"

    portfolio = MPT(tickers=tckers, startdate=start, enddate=end)

    portfolio.print_sharpe_info()
    portfolio.display_results()


if __name__ == "__main__":
    main()




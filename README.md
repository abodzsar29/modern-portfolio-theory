# Modern Portfolio Theory – Efficient Frontier Optimisation

A **Python** implementation of **Modern Portfolio Theory (MPT)** that constructs an optimally weighted equity portfolio by maximising the **Sharpe ratio** and visualising the **Efficient Frontier**.

Historical price data is fetched automatically from Yahoo Finance, and the optimisation is performed using the **PyPortfolioOpt** library backed by **CVXPY**.

---

## Overview

Modern Portfolio Theory, introduced by Harry Markowitz in 1952, provides a mathematical framework for assembling a portfolio of assets such that the expected return is maximised for a given level of risk. This project operationalises that framework for any user-specified basket of equities and date range.

The core `MPT` class encapsulates the full workflow:

1. **Data retrieval** – downloads adjusted closing prices via `yfinance`.
2. **Return estimation** – computes annualised geometric mean returns.
3. **Risk modelling** – constructs a sample covariance matrix.
4. **Optimisation** – finds the **tangency portfolio** (maximum Sharpe ratio) subject to long-only weight constraints (no short selling).
5. **Reporting** – prints individual stock weightings and summarised portfolio performance metrics.
6. **Visualisation** – plots the Efficient Frontier with the tangency portfolio highlighted.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical market data retrieval |
| `PyPortfolioOpt` | Return estimation, risk modelling, and portfolio optimisation |
| `cvxpy` | Convex optimisation backend |
| `numpy` | Numerical operations |
| `pandas` | Tabular data handling |
| `matplotlib` | Efficient Frontier visualisation |

Install all dependencies via pip:

```bash
pip install yfinance pypfopt cvxpy numpy pandas matplotlib
```

---

## Usage

### Running the default example

The `main()` function constructs a portfolio of **20 large-cap US equities** over the calendar year **2023**:

```bash
python mpt.py
```

This will:
- Print the optimal weight for each stock in the portfolio.
- Print the expected annual return, annual volatility, and Sharpe ratio.
- Display a plot of the Efficient Frontier with the tangency portfolio marked.

### Using the `MPT` class directly

```python
from mpt import MPT

tickers = "AAPL MSFT GOOGL AMZN NVDA"
portfolio = MPT(tickers=tickers, startdate="2022-01-01", enddate="2023-12-31")

# Print Sharpe ratio optimisation results
portfolio.print_sharpe_info()

# Display the Efficient Frontier plot
portfolio.display_results()
```

---

## Class Reference

### `MPT(tickers, startdate, enddate)`

| Parameter | Type | Description |
|---|---|---|
| `tickers` | `str` | Space-separated list of ticker symbols (e.g. `"AAPL MSFT GOOGL"`) |
| `startdate` | `str` | Start date for historical data in `YYYY-MM-DD` format |
| `enddate` | `str` | End date for historical data in `YYYY-MM-DD` format |

#### Public Methods

| Method | Description |
|---|---|
| `print_sharpe_info()` | Prints per-stock weights and overall portfolio performance metrics |
| `display_results()` | Plots the Efficient Frontier and marks the tangency portfolio |

---

## Output Example

```
Individual Stock Weightings in Portfolio:
PEP: 0.0312
ADBE: 0.0000
...
#####################
Portfolio Performance Summarised:
Expected annual return: 18.4%
Annual volatility: 12.1%
Sharpe Ratio: 1.37
```

The Efficient Frontier plot displays all assets as scatter points, with the optimal (maximum Sharpe ratio) portfolio marked by a red star (★).

---

## Constraints

- **Long-only**: all portfolio weights are bounded to `[0, 1]`, precluding short positions.
- **Fully invested**: the optimisation enforces that weights sum to 1 (i.e. 100% of the portfolio is allocated).
- Returns are estimated using the **compounding geometric mean** of historical adjusted close prices.
- Risk is modelled using the **sample covariance matrix**.

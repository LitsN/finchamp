# World-ETF Investment Dashboard

The dashboard is an interactive learning to understand the performance and risk of a world ETF compared with tactical investment
Live App: https://finchamp.streamlit.app/

## Key Features
Analyze and compare different investment scenarios:
- Longterm performance and risk of a World ETF
- World ETF vs stock picking
- World ETF vs gold investment
- World ETF mixed with gold
- World ETF vs World-Funds
- World ETF: Buy and Hold vs Buy the Dip

## Risk Analysis
The app implements two core methods to evaluate absolute performance and risk:
- Backtest of historical data using different rolling windows
- MonteCarlo simulation and MarkovChain using historical yield and volatility, different rolling windows

## Tech Stack
- Frontend: Streamlit
- Data Analysis, backtest and simulation: pandas, numpy
- Financial Data: yfinance
- Visualization: plotly

## Author
Developed by **Nico Litschke**

*Questions, feedback or bugs? Reach out via www.linkedin.com/in/nico-litschke or visit www.nicolitschke.com*

# Inflation Forecast

A Python-based econometrics research project forecasting U.S. inflation using macroeconomic time series data from the FRED API.

## Research Question
*Can lagged macroeconomic variables (unemployment, money supply, oil prices, federal funds rate) be used to forecast U.S. CPI inflation?*

## Project Goals
1. Pull and clean macroeconomic data from FRED
2. Perform exploratory data analysis (EDA)
3. Run stationarity tests and transform data as needed
4. Build ARIMA and VAR forecasting models
5. Evaluate model performance using RMSE and MAE
6. Build an interactive Streamlit dashboard
7. Write a research paper documenting methodology and findings

## Data Sources (FRED)
| Variable | Series ID | Description |
|---|---|---|
| CPI | CPIAUCSL | Consumer Price Index |
| Unemployment | UNRATE | Unemployment Rate |
| M2 | M2SL | M2 Money Supply |
| Oil | DCOILWTICO | WTI Crude Oil Price |
| Fed Funds | FEDFUNDS | Federal Funds Rate |

## Tech Stack
- **Data:** pandas, fredapi
- **Modeling:** statsmodels, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Dashboard:** Streamlit
- **Environment:** Python 3.10, python-dotenv

## Project Structure
```
inflation-forecast/
├── data/           # Raw + cleaned CSVs from FRED
├── notebooks/      # Jupyter notebooks for EDA and modeling
├── src/            # Python scripts (pipeline, models)
├── dashboard/      # Streamlit app
├── paper/          # Research writeup
├── docs/           # Project documentation
├── .env            # FRED API key (never committed)
├── .gitignore
└── requirements.txt
```

## Setup
```bash
# Clone the repo
git clone https://github.com/ACKibler/inflation-forecast.git
cd inflation-forecast

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your FRED API key to .env
echo "FRED_API_KEY=your_key_here" > .env

# Run the data pipeline
python src/fetch_data.py
```

## Status
See [docs/PROGRESS.md](docs/PROGRESS.md) for the latest updates.
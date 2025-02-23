# PyPort
Modern Portfolio Optimization 

# Upcoming 
- [x] OU mean reversion and multi-asset statistical arbitrage
![image](https://github.com/user-attachments/assets/e838a18d-7173-4102-91eb-40da5c8a96dd)
![image](https://github.com/user-attachments/assets/c9b31d53-ef5e-46c9-a39f-e04aea618fe8)
- [x] Return-risk-sharpe profile of portfolio assets to see how much they contribute
- [x] New unified strategy optimizers with vol and CVaR constraints
- [ ] DuckDB migration for faster data loading

# February 2025 
- Updated features with more modern techniques: (VAE for anomaly detection would be considered more SOTA, but due to our data sample size and the marginal benefit vs computational resources required, isolation forest was selected.
- Changed the data processing to use the full available history for various computations vs slicing to the latest data range available among newer assets first.
- Fixed mean reversion hyperparameters optimization study results and continuous vs discrete signal generation for weight adjustment vs binary exclusion/inclusion
- Balanced precision for computational speed by using clustering to apply group optimal params vs for each individual asset for fast results
- Overall the results are very similar to the previous version which is a good sign for consistency!

- [x] Anomaly detection with Isolation Forest
![image](https://github.com/user-attachments/assets/bab5d481-98eb-441a-9f6a-aea95d30b204)
- [x] HDBSCAN clustering for de-correlation
![image](https://github.com/user-attachments/assets/2187106b-dad4-49cb-a005-597f50c9b089)
- [x] Robust Z-score Mean Reversion with Volatility-Adaptive Parameters and Stateful Signals
![image](https://github.com/user-attachments/assets/1c4a3c0f-6d34-4e2b-b8b4-66fa58e43bef)


# January 2025
- Reworked implementation of Nested Clustering (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961) with vector ops => much faster!
- Added anomaly detection with Kalman Filter and mean reversion filters
- Integrated Optuna to dynamically find optimal thresholds for maximizing performance metrics and returns 

- Daily and Cumulative Returns Statistics
- ![image](https://github.com/user-attachments/assets/ab6fd50c-f9ae-49ff-a12c-b1526df83239)

- Anomaly Detection for Automatic Filtering
- Hierarchical Clustering for De-correlation
- Dynamic Z-score Thresholds for Mean Reversion
- All Automated Hyperparameter Tuning

## Instructions

### Setting Up Configuration
- Create a `config.yaml` file based on the provided `testconfig.yaml` template.
- In the `config.yaml` file, you can specify:
  - Input files containing ticker symbols.
  - Models and time periods to optimize against.
  - Whether to use various filters like de-correlation.

### Preparing Input Files
- Each input file (CSV format) should have a single column containing the ticker symbols to include in the optimization.
- Any commented lines (starting with `#`) will be ignored during processing.

### Configuring Input Files in `config.yaml`
- Under the `input_files` section of the config, list multiple files if needed. For example:
  ```yaml
  input_files:
    - file1.csv
    - file2.csv

You can specify multiple models under models. The output will contain a simple average of all selected models and time periods.

# Local Usage

This project is now compatible with LTS Python version as of January 2025 (3.12.8) and can be installed simply with 
```
pip install -r requirements.txt
```
There is a CLI main runner, API main available, as well as a new iterative_pipeline for entrypoints.

# Running the API

The API allows you to execute the pipeline dynamically by providing configurations and symbol overrides.

## Starting the API

Run the following command to start the API server:

```
python api_main.py
```

The server will start on `http://localhost:8000`.

## API Endpoint

**POST** `/inference`

- **Description**: Runs the portfolio optimization pipeline.
- **Payload**: json { "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"] }

- **Example Request**:

```
curl -X POST "http://localhost:8000/inference" \
-H "Content-Type: application/json" \
-d '{
  "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"],
}'
```

- **Response**: A JSON object containing the results of the pipeline:

```json
{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "models": "model_name_1, model_name_2",
  "symbols": ["symbol1", "symbol2"],
  "normalized_avg": { "symbol1": 0.25, "symbol2": 0.75 },
  "daily_returns": pd.DataFrame,
  "cumulative_returns": pd.DataFrame
}
```

---

# Notes

- Ensure your `config.yaml` file is correctly configured for your use case.
- The API allows dynamic overrides for symbols and configurations without modifying the local `config.yaml`.
- The symbol selection process will optionally filter out inputs based on a number of configurable methods prior to optimization so the result set may not contain all assets in the input list. Enabled by default.
- SIM_PORT is the hypothetical portfolio with the allocation result over the same period (used for graphing daily and cumulative returns).



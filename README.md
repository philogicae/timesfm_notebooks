# timesfm_notebooks

TimesFM 2.5-200M forecasting notebooks — a minimal template and a full crypto example.

## Notebooks

### Template — [`timesfm.ipynb`](timesfm.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philogicae/timesfm_notebooks/blob/main/timesfm.ipynb)

Minimal starter. Loads the model, forecasts two dummy series (`np.linspace`, `np.sin`), inspects the output shapes, then plots inputs vs forecasts and saves a PNG. Fork this to build your own pipeline — five cells, no data source, no transforms. Just the model, a forecast, and a chart.

### Crypto — [`timesfm_crypto.ipynb`](timesfm_crypto.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philogicae/timesfm_notebooks/blob/main/timesfm_crypto.ipynb)

Full crypto price forecasting example. A residual-space model: log returns are normalized by rolling volatility, forecast as a series by TimesFM, then denormalized back to a price path with a confidence band.

Open in Google Colab (T4 GPU, configured by default) or Jupyter, then run all cells. The model loads **once** (early cell, slow) and persists in the kernel while you iterate on the rest. A final **Run** cell executes the pipeline with timed logs:

```
[1/3] Loaded 2 series: BTC-USD, ETH-USD (2.1s)
[2/3] Preparing + inference...
      done (8.4s)
[3/3] Rendering chart...
✓ total 10.7s
```

#### Layout

Three sections, top to bottom.

| Section | Block | Role |
|---------|-------|------|
| **1 · Global** | Dependencies & utilities | inline `pip install`, imports, color palette/helpers |
| | Model | `load_model()` — loads `google/timesfm-2.5-200m-pytorch` once, compiles with `ForecastConfig` |
| | Inference | `run_inference(data, model, params)` — forecasts the `Signal` column, returns `{med, lo, hi}` |
| | Display | `plot_results(data, results, params)` — log-scale dual-axis Plotly chart |
| **2 · Data** | Parameters | the `PARAMS` dict |
| | Sources | `from_yfinance` / `from_csv` / `from_dict` — **replace this to fork** |
| | Utils | `prepare_series` / `prepare_all` / `to_levels` — residual-space transform |
| **3 · Execution** | Run | pick a source, then prepare → infer → denormalize → plot |

#### Forking the data source

The downstream cells depend only on a small contract, not on `yfinance` or crypto specifically. A source returns `dict[str, pd.Series]` keyed by series name, each with a `DatetimeIndex`. `prepare_all` then turns each series into a `pd.DataFrame` with columns:

- `Level` — resampled level series (anchors the forecast to the last observed value)
- `Vol` — trailing 20-window rolling volatility (denormalizes the residual forecast back to level)
- `Signal` — normalized residuals `log_return / vol` (what TimesFM sees)

Swap the **Sources** block for a CSV load, synthetic generator, or another API (three are already provided), add the new series to `PARAMS` (`symbols`, `colors`), and everything downstream still works.

#### Forecast

- **Model:** `google/timesfm-2.5-200m-pytorch`, compiled with `max_context=1024`, `max_horizon=256`, normalized inputs, continuous quantile head, flip invariance, positive output, fixed quantile crossing
- **Context:** up to 1024 steps (~8.5 years) of 3-day bars
- **Horizon:** 250 steps (~750 days)
- **Signal pipeline:** resample (pchip) → log returns → rolling volatility → normalized residuals → TimesFM → denormalized price path
- **Quantiles:** median (q=5) with a ±0.5 band (q=4.5 / q=5.5), linearly interpolated between the model's 10 quantile heads
- **Denormalization:** `last * exp(cumsum(forecast * vol))`
- **Visualization:** log-scale dual-axis Plotly chart (first series left, others right), dashed forecast median, shaded band, guide line at the forecast start, x-axis rangeslider

#### Parameters

All knobs live in the `PARAMS` dict in the **Data · Parameters** cell:

```python
PARAMS = {
    'symbols': ['BTC-USD', 'ETH-USD'],
    'resample_rule': '3D',
    'horizon': 250,
    'max_context': 1024,
    'colors': {'BTC-USD': '#F7931A', 'ETH-USD': '#627EEA'},
}
```

## Dependencies

Installed inline in each notebook: `timesfm`, plus `yfinance` / `plotly` / `pandas` / `numpy` / `kaleido` in the crypto notebook and `matplotlib` in the template. Model weights are pulled from Hugging Face at runtime. GPU is enabled by default via notebook metadata (`accelerator: GPU`, Colab `gpuType: T4`, Python 3.10).

Both notebooks save a PNG of their chart on each run (`timesfm_crypto.png`, `timesfm.png`) and display it in a final cell, so the figure persists in the notebook output and renders on GitHub without re-executing.

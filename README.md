# timesfm_notebooks

Forecast any time series with Google's [TimesFM](https://github.com/google-research/timesfm) 2.5-200M — median path with a ±0.5-quantile confidence band.

<p align="center">
  <img src="cover.png" alt="TimesFM forecast chart" width="720">
</p>

## Notebooks

|                                                                                                                                                                            | Notebook                                       | Data                         | Horizon     |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------- | ---------------------------- | ----------- |
|    [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philogicae/timesfm_notebooks/blob/main/timesfm.ipynb)     | [`timesfm.ipynb`](timesfm.ipynb)               | noisy sine + sawtooth        | 12          |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philogicae/timesfm_notebooks/blob/main/timesfm_crypto.ipynb) | [`timesfm_crypto.ipynb`](timesfm_crypto.ipynb) | `BTC` / `ETH` via `yfinance` | 250 (~750d) |

Both share the same 22-cell pipeline (**Global** · **Data** · **Execution**) — only the Data section differs. The template uses an identity transform; the crypto notebook uses residual-space (`Signal = log_return / vol`, `to_levels = last * exp(cumsum(fcst * vol))`).

---

## Quick start

1. Click **Open in Colab** on `timesfm.ipynb` above
2. Bottom-center → click the **Gemini icon** → select **Gemini 3 Flash**
3. Tell it what to forecast. It fetches the data, edits the Sources cell, runs the notebook.

   **Examples:**
   - **Weather** — _"Pull hourly temperature for Tokyo last year from Open-Meteo and forecast the next 2 weeks"_
   - **Sales** — _"Load my Q1–Q3 daily sales CSV from `/content/sales.csv` and forecast Q4"_
   - **Web traffic** — _"Fetch daily pageviews for wikipedia.org from the Wikimedia API and forecast 30 days"_

4. Chart + confidence band renders inline · PNG saved for GitHub

> ⚠️ Works great on most time series. Crypto is hard — the `timesfm_crypto.ipynb` example is a demo, not financial advice.

<details>
<summary><b>Manual run</b></summary>

Open in Colab (T4 GPU, preconfigured) → **Run all**. Model loads once, final cell runs the pipeline:

```
[1/3] Loaded 2 series: BTC-USD, ETH-USD (0.4s)
[2/3] Preparing + inference... done (0.7s)
[3/3] Rendering chart... ✓ total 6.9s
```

Chart PNG is saved and embedded in the notebook, so it renders on GitHub without re-executing.

</details>

---

## Fork

A source returns `dict[str, pd.Series]` with a `DatetimeIndex`. `prepare_all` turns each into a `DataFrame` with `Level` (history) + `Signal` (what TimesFM sees). Swap the **Sources** block, add series to `PARAMS`, rest runs unchanged.

```python
PARAMS = {
    'symbols': [...],
    'resample_rule': '3D',
    'horizon': 250,
    'colors': {...},
    'labels': {...},
}
```

## Details

- **Model** — `google/timesfm-2.5-200m-pytorch` · `max_context=1024` · `max_horizon=256`
- **Quantiles** — median (q=5) ± 0.5 band (q=4.5 / q=5.5), interpolated across 10 heads
- **Chart** — 1280×720 log-scale dual-axis Plotly, custom axis labels, dashed median, shaded band, rangeslider
- **Deps** — `timesfm`, `plotly>=6.1.1`, `plotly-geo`, `kaleido`, `pandas`, `numpy` + Chrome/kaleido headless setup (inline) · crypto adds `yfinance`
- **Runtime** — HuggingFace weights at runtime · GPU via notebook metadata (`gpuType: T4`, Python 3.10)

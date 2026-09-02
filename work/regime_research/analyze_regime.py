from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
CHARTS = OUT / "charts"
for folder in (OUT, DATA, CHARTS):
    folder.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels
import yfinance as yf


AS_OF_REQUESTED = pd.Timestamp("2026-09-02")
DOWNLOAD_START = "2021-01-01"
DOWNLOAD_END_EXCLUSIVE = "2026-09-03"
LOOKBACK = 252
BASE_TC_BP = 10.0
BASE_BORROW = 0.01

UNIVERSE = {
    "6981.T": ("村田製作所", "受動部品・MLCC"),
    "6976.T": ("太陽誘電", "受動部品・MLCC"),
    "6762.T": ("TDK", "受動部品・センサ"),
    "6971.T": ("京セラ", "電子部品・セラミック"),
    "6770.T": ("アルプスアルパイン", "電子部品・センサ"),
    "6806.T": ("ヒロセ電機", "コネクタ"),
    "6807.T": ("日本航空電子工業", "コネクタ"),
    "6963.T": ("ローム", "半導体・電子部品"),
    "6988.T": ("日東電工", "電子材料"),
    "6479.T": ("ミネベアミツミ", "精密・電子部品"),
    "4980.T": ("デクセリアルズ", "電子材料"),
    "6817.T": ("スミダコーポレーション", "受動部品・コイル"),
    "6779.T": ("日本電波工業", "受動部品・水晶"),
    "6997.T": ("日本ケミコン", "受動部品・コンデンサ"),
}
PAIR = ("6981.T", "6976.T")

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
]
for font_path in FONT_PATHS:
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
        break
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F9FC",
        "axes.edgecolor": "#9AA6B2",
        "axes.grid": True,
        "grid.color": "#DDE4EA",
        "grid.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)

NAVY = "#173B57"
BLUE = "#1479B8"
ORANGE = "#E38B2C"
RED = "#C44E52"
GREEN = "#2E8B57"
PURPLE = "#6657A3"
GRAY = "#6B7785"


@dataclass(frozen=True)
class Params:
    entry_z: float
    exit_z: float
    max_hold: int
    stop_z: float = 4.0


LOW = Params(1.0, 0.5, 20, 4.0)
CONSERVATIVE = Params(2.0, 0.5, 60, 4.0)


def fetch_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    close: dict[str, pd.Series] = {}
    volume: dict[str, pd.Series] = {}
    metadata: list[dict[str, object]] = []
    for ticker, (name, subsector) in UNIVERSE.items():
        hist = yf.Ticker(ticker).history(
            start=DOWNLOAD_START,
            end=DOWNLOAD_END_EXCLUSIVE,
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=False,
        )
        if hist.empty:
            raise RuntimeError(f"Yahoo Finance returned no rows for {ticker}")
        idx = pd.to_datetime(hist.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        hist.index = idx.normalize()
        if "Adj Close" not in hist or "Volume" not in hist:
            raise RuntimeError(f"Missing adjusted close/volume for {ticker}")
        close[ticker] = hist["Adj Close"].rename(ticker)
        volume[ticker] = hist["Volume"].rename(ticker)
        hist.to_csv(DATA / f"yahoo_{ticker.replace('.', '_')}_raw.csv", encoding="utf-8-sig")
        metadata.append(
            {
                "ticker": ticker,
                "name": name,
                "subsector": subsector,
                "first_date": str(hist.index.min().date()),
                "last_date": str(hist.index.max().date()),
                "rows": int(len(hist)),
                "source": f"Yahoo Finance ({ticker})",
                "access_date": "2026-09-02",
            }
        )
    px = pd.concat(close.values(), axis=1).sort_index()
    vol = pd.concat(volume.values(), axis=1).sort_index()
    # Yahoo's live 2026-09-01 row was present but its Close/Adj Close fields were
    # temporarily blank on the 2026-09-02 refresh. Preserve the already captured
    # Yahoo adjusted closes from the preceding report for the two target legs only.
    prior_pair_file = ROOT / "work" / "adjusted_prices.csv"
    if prior_pair_file.exists():
        prior = pd.read_csv(prior_pair_file, parse_dates=["Date"]).set_index("Date")
        for ticker, old_col in [(PAIR[0], "murata"), (PAIR[1], "taiyo")]:
            if old_col in prior:
                px[ticker] = px[ticker].combine_first(prior[old_col])
    px.index.name = "Date"
    vol.index.name = "Date"
    px.to_csv(DATA / "universe_adjusted_close.csv", encoding="utf-8-sig")
    vol.to_csv(DATA / "universe_volume.csv", encoding="utf-8-sig")
    pd.DataFrame(metadata).to_csv(OUT / "universe.csv", index=False, encoding="utf-8-sig")
    return px, vol


def rolling_pair_model(prices: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    logs = np.log(prices[list(PAIR)].dropna())
    rows: list[dict[str, object]] = []
    for i in range(lookback, len(logs)):
        train = logs.iloc[i - lookback : i]
        x = train[PAIR[1]].to_numpy()
        y = train[PAIR[0]].to_numpy()
        beta, alpha = np.polyfit(x, y, 1)
        resid = y - (alpha + beta * x)
        sd = float(np.std(resid, ddof=1))
        spread = float(logs[PAIR[0]].iloc[i] - (alpha + beta * logs[PAIR[1]].iloc[i]))
        z = (spread - float(resid.mean())) / sd if sd > 0 else np.nan
        rows.append(
            {
                "Date": logs.index[i],
                "alpha": float(alpha),
                "beta": float(beta),
                "spread": spread,
                "train_mean": float(resid.mean()),
                "train_std": sd,
                "z": float(z),
            }
        )
    return pd.DataFrame(rows).set_index("Date")


def signal_state(z: pd.Series, params: Params, entry_allowed: pd.Series | None = None) -> pd.DataFrame:
    state = 0
    held = 0
    states: list[int] = []
    reasons: list[str] = []
    if entry_allowed is None:
        allowed_values = np.ones(len(z), dtype=bool)
    else:
        allowed_values = entry_allowed.reindex(z.index).fillna(False).to_numpy(dtype=bool)
    for value, allowed in zip(z.to_numpy(), allowed_values):
        reason = "hold"
        if not np.isfinite(value):
            state, held, reason = 0, 0, "invalid"
        elif state == 0:
            if not allowed:
                reason = "gate_off"
            elif value <= -params.entry_z:
                state, held, reason = 1, 0, "enter_long_spread"
            elif value >= params.entry_z:
                state, held, reason = -1, 0, "enter_short_spread"
            else:
                reason = "flat"
        else:
            held += 1
            stop = (state == 1 and value <= -params.stop_z) or (state == -1 and value >= params.stop_z)
            exit_ = (state == 1 and value >= -params.exit_z) or (state == -1 and value <= params.exit_z)
            if stop:
                state, held, reason = 0, 0, "stop"
            elif held >= params.max_hold:
                state, held, reason = 0, 0, "max_hold"
            elif exit_:
                state, held, reason = 0, 0, "mean_reversion_exit"
        states.append(state)
        reasons.append(reason)
    return pd.DataFrame({"target_dir": states, "reason": reasons}, index=z.index)


def extract_trades(daily: pd.DataFrame) -> pd.DataFrame:
    dirs = daily["position_dir"].astype(int)
    records: list[dict[str, object]] = []
    start: pd.Timestamp | None = None
    direction = 0
    for i, (dt, current) in enumerate(dirs.items()):
        prev = int(dirs.iloc[i - 1]) if i else 0
        if prev == 0 and current != 0:
            start = dt
            direction = int(current)
        if prev != 0 and current == 0 and start is not None:
            sl = daily.loc[start:dt, "net_return"]
            records.append(
                {
                    "entry_date": start,
                    "exit_date": dt,
                    "direction": "long_spread" if direction == 1 else "short_spread",
                    "holding_days": max(len(sl) - 1, 0),
                    "trade_return": float((1 + sl).prod() - 1),
                }
            )
            start = None
            direction = 0
    return pd.DataFrame(records)


def metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int]:
    r = daily["net_return"].fillna(0.0)
    eq = (1 + r).cumprod()
    n = len(r)
    years = n / 252.0
    vol_d = float(r.std(ddof=1)) if n > 1 else np.nan
    dd = eq / eq.cummax() - 1
    return {
        "total_return": float(eq.iloc[-1] - 1) if n else np.nan,
        "ann_return": float(eq.iloc[-1] ** (1 / years) - 1) if n and years > 0 else np.nan,
        "ann_vol": float(vol_d * math.sqrt(252)) if np.isfinite(vol_d) else np.nan,
        "sharpe": float(r.mean() / vol_d * math.sqrt(252)) if vol_d > 0 else np.nan,
        "max_drawdown": float(dd.min()) if n else np.nan,
        "trades": int(len(trades)),
        "win_rate": float((trades["trade_return"] > 0).mean()) if len(trades) else np.nan,
        "avg_holding_days": float(trades["holding_days"].mean()) if len(trades) else np.nan,
        "turnover": float(daily["turnover"].sum()),
    }


def run_backtest(
    prices: pd.DataFrame,
    model: pd.DataFrame,
    params: Params,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tc_bp: float = BASE_TC_BP,
    borrow_rate: float = BASE_BORROW,
    gate: pd.Series | None = None,
    liquidate: bool = True,
) -> dict[str, object]:
    sm = model.loc[start:end].copy()
    allowed: pd.Series | None = None
    if gate is not None:
        allowed = gate.reindex(sm.index).ffill().fillna(False).astype(bool)
    # The gate controls entry only.  Once a position exists, normal z/stop/time
    # exits remain in force; ordinary gate-off is not an emergency liquidation.
    sig = signal_state(sm["z"], params, allowed)
    direction = sig["target_dir"].shift(1).fillna(0).astype(int)
    if liquidate and len(direction):
        direction.iloc[-1] = 0
    w_m = direction.astype(float) * 0.5
    w_t = direction.astype(float) * -0.5
    aligned = prices[list(PAIR)].reindex(sm.index)
    asset_r = aligned.pct_change()
    gross = w_m.shift(1).fillna(0) * asset_r[PAIR[0]].fillna(0)
    gross += w_t.shift(1).fillna(0) * asset_r[PAIR[1]].fillna(0)
    turnover = w_m.diff().abs().fillna(w_m.abs()) + w_t.diff().abs().fillna(w_t.abs())
    short_prev = (-w_m.shift(1).fillna(0)).clip(lower=0) + (-w_t.shift(1).fillna(0)).clip(lower=0)
    tc = turnover * tc_bp / 10000
    borrow = short_prev * borrow_rate / 252
    net = gross - tc - borrow
    daily = pd.DataFrame(
        {
            "z": sm["z"],
            "beta": sm["beta"],
            "target_dir": sig["target_dir"],
            "position_dir": direction,
            "w_murata": w_m,
            "w_taiyo": w_t,
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": tc,
            "borrow_cost": borrow,
            "net_return": net,
        }
    )
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    tr = extract_trades(daily)
    return {"daily": daily, "trades": tr, "metrics": metrics(daily, tr)}


def half_life(residual: pd.Series | np.ndarray) -> float:
    r = pd.Series(np.asarray(residual, dtype=float)).dropna()
    if len(r) < 20:
        return np.nan
    lag = r.shift(1).dropna()
    delta = r.diff().dropna().reindex(lag.index)
    slope, _ = np.polyfit(lag.to_numpy(), delta.to_numpy(), 1)
    if slope >= 0:
        return np.inf
    return float(-math.log(2) / slope)


def beta_stability(logs: pd.DataFrame) -> tuple[float, float]:
    betas: list[float] = []
    for k in range(4):
        block = logs.iloc[-(k + 1) * 63 : -k * 63 if k else None]
        if len(block) < 50:
            continue
        beta, _ = np.polyfit(block.iloc[:, 1].to_numpy(), block.iloc[:, 0].to_numpy(), 1)
        betas.append(float(beta))
    if not betas:
        return np.nan, np.nan
    median = float(np.median(betas))
    cv = float(np.std(betas, ddof=1) / max(abs(median), 0.05)) if len(betas) > 1 else 0.0
    return median, cv


def pair_diagnostics(prices: pd.DataFrame, model: pd.DataFrame, low_daily: pd.DataFrame) -> pd.DataFrame:
    logs = np.log(prices[list(PAIR)].dropna())
    rows: list[dict[str, object]] = []
    # Use the last actually observed session in each Friday-ending week.  This
    # avoids labeling a Tuesday observation with a future Friday date.
    weekly_dates = pd.DatetimeIndex(
        model.index.to_series().groupby(model.index.to_period("W-FRI")).max().to_list()
    )
    for period_end in weekly_dates:
        available = logs.loc[:period_end]
        if len(available) < 252:
            continue
        w252 = available.iloc[-252:]
        w126 = available.iloc[-126:]
        beta, alpha = np.polyfit(w252[PAIR[1]], w252[PAIR[0]], 1)
        resid = w252[PAIR[0]] - (alpha + beta * w252[PAIR[1]])
        resid126 = resid.iloc[-126:]
        sd = float(resid.std(ddof=1))
        z = (resid - resid.mean()) / sd if sd > 0 else resid * np.nan
        corr = float(w126.pct_change().corr().iloc[0, 1])
        _, bcv = beta_stability(w252)
        try:
            adf_p = float(adfuller(resid, regression="c", autolag="AIC", result_object=False)[1])
        except Exception:
            adf_p = np.nan
        try:
            eg_p = float(coint(w252.iloc[:, 0], w252.iloc[:, 1], trend="c", autolag="aic")[1])
        except Exception:
            eg_p = np.nan
        zero_cross = int((np.sign(resid126 - resid126.mean()).diff().abs() == 2).sum())
        opp63 = int((z.iloc[-63:].abs() >= 1.0).astype(int).diff().eq(1).sum())
        hist = low_daily.loc[:period_end].iloc[-126:]
        hist_trades = extract_trades(hist) if len(hist) else pd.DataFrame()
        hret = hist["net_return"] if len(hist) else pd.Series(dtype=float)
        hvol = float(hret.std(ddof=1)) if len(hret) > 1 else np.nan
        hsh = float(hret.mean() / hvol * math.sqrt(252)) if hvol > 0 else np.nan
        rows.append(
            {
                "Date": period_end,
                "corr126": corr,
                "beta252": float(beta),
                "beta_cv": bcv,
                "adf_p252": adf_p,
                "eg_p252": eg_p,
                "half_life252": half_life(resid),
                "resid_sigma252": sd,
                "zero_cross126": zero_cross,
                "opportunities63": opp63,
                "trail126_sharpe": hsh,
                "trail126_return": float((1 + hret).prod() - 1) if len(hret) else np.nan,
                "trail126_trades": int(len(hist_trades)),
                "trail126_win_rate": float((hist_trades["trade_return"] > 0).mean()) if len(hist_trades) else np.nan,
            }
        )
    d = pd.DataFrame(rows).set_index("Date")
    # Frozen transparent gates; thresholds were specified without inspecting 2026 outcomes.
    raw_s = (
        (d["corr126"] >= 0.45)
        & (d["beta252"] > 0)
        & (d["beta_cv"] <= 0.75)
        & d["half_life252"].between(2, 40)
        & (d["adf_p252"] <= 0.20)
        & (d["zero_cross126"] >= 4)
        & (d["opportunities63"] >= 2)
    )
    raw_p = (
        raw_s
        & (d["trail126_sharpe"] > 0)
        & (d["trail126_trades"] >= 3)
        & (d["trail126_win_rate"] >= 0.50)
    )
    d["gate_structural"] = raw_s.rolling(2).sum().eq(2)
    d["gate_confirmed"] = raw_p.rolling(2).sum().eq(2)
    return d


def expand_weekly_gate(index: pd.Index, weekly: pd.Series) -> pd.Series:
    s = weekly.astype(bool).reindex(index).ffill().fillna(False)
    return s.astype(bool)


def change_point_one_break(series: pd.Series, min_segment: int) -> dict[str, object]:
    s = series.dropna().astype(float)
    values = s.to_numpy()
    n = len(values)
    if n < 2 * min_segment + 1:
        return {"min_segment": min_segment, "date": None, "bic_improvement": np.nan}
    sse0 = float(((values - values.mean()) ** 2).sum())
    best: tuple[float, int] | None = None
    for k in range(min_segment, n - min_segment):
        left, right = values[:k], values[k:]
        sse = float(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k)
    assert best is not None
    sse1, k = best
    # Gaussian BIC, one mean vs two means (variance common); positive means break model is preferred.
    bic0 = n * math.log(max(sse0 / n, 1e-14)) + 2 * math.log(n)
    bic1 = n * math.log(max(sse1 / n, 1e-14)) + 3 * math.log(n)
    return {
        "min_segment": min_segment,
        "date": str(s.index[k].date()),
        "bic_improvement": float(bic0 - bic1),
        "mean_before": float(values[:k].mean()),
        "mean_after": float(values[k:].mean()),
        "ann_mean_before": float(values[:k].mean() * 52),
        "ann_mean_after": float(values[k:].mean() * 52),
    }


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float)
    valid = p.dropna().sort_values()
    m = len(valid)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if not m:
        return q
    raw = valid.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    q.loc[valid.index] = np.clip(adjusted, 0, 1)
    return q


def economic_link(sub1: str, sub2: str) -> float:
    if sub1 == sub2:
        return 1.0
    passive1 = sub1.startswith("受動部品")
    passive2 = sub2.startswith("受動部品")
    if passive1 and passive2:
        return 0.85
    if any(x in sub1 for x in ("電子部品", "電子材料", "半導体")) and any(
        x in sub2 for x in ("電子部品", "電子材料", "半導体")
    ):
        return 0.65
    return 0.45


def static_validation(z: pd.Series, r1: pd.Series, r2: pd.Series, tc_bp: float = 10.0) -> dict[str, float]:
    sig = signal_state(z, LOW)
    pos = sig["target_dir"].shift(1).fillna(0).astype(int)
    if len(pos):
        pos.iloc[-1] = 0
    w1, w2 = 0.5 * pos, -0.5 * pos
    gross = w1.shift(1).fillna(0) * r1.fillna(0) + w2.shift(1).fillna(0) * r2.fillna(0)
    turnover = w1.diff().abs().fillna(w1.abs()) + w2.diff().abs().fillna(w2.abs())
    short = (-w1.shift(1).fillna(0)).clip(lower=0) + (-w2.shift(1).fillna(0)).clip(lower=0)
    net = gross - turnover * tc_bp / 10000 - short * 0.01 / 252
    tmp = pd.DataFrame({"position_dir": pos, "net_return": net})
    tr = extract_trades(tmp)
    sd = float(net.std(ddof=1))
    return {
        "val_return": float((1 + net).prod() - 1),
        "val_sharpe": float(net.mean() / sd * math.sqrt(252)) if sd > 0 else np.nan,
        "val_trades": int(len(tr)),
        "val_win_rate": float((tr["trade_return"] > 0).mean()) if len(tr) else np.nan,
    }


def screen_monthly(px: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    observed = px.loc["2023-01-01":].dropna(how="all").index
    # Month-end screen date is the final actually observed session, never the
    # calendar month-end label produced by resample.
    dates = pd.DatetimeIndex(
        observed.to_series().groupby(observed.to_period("M")).max().to_list()
    )
    rows: list[dict[str, object]] = []
    tickers = list(UNIVERSE)
    for dt in dates:
        month_rows: list[dict[str, object]] = []
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1 :]:
                p = px.loc[:dt, [t1, t2]].dropna().iloc[-378:]
                if len(p) < 350:
                    continue
                logs = np.log(p)
                train = logs.iloc[:-126]
                val = logs.iloc[-126:]
                beta, alpha = np.polyfit(train[t2], train[t1], 1)
                train_resid = train[t1] - (alpha + beta * train[t2])
                sd = float(train_resid.std(ddof=1))
                z_val = (val[t1] - (alpha + beta * val[t2]) - train_resid.mean()) / sd if sd > 0 else val[t1] * np.nan
                ret = p.pct_change()
                corr = float(ret.iloc[-252:].corr().iloc[0, 1])
                try:
                    eg_p = float(coint(logs.iloc[-252:, 0], logs.iloc[-252:, 1], trend="c", autolag="aic")[1])
                except Exception:
                    eg_p = np.nan
                full_beta, full_alpha = np.polyfit(logs[t2].iloc[-252:], logs[t1].iloc[-252:], 1)
                resid = logs[t1].iloc[-252:] - (full_alpha + full_beta * logs[t2].iloc[-252:])
                hl = half_life(resid)
                _, bcv = beta_stability(logs.iloc[-252:])
                normalized = p.iloc[-252:] / p.iloc[-252]
                distance = float(((normalized[t1] - normalized[t2]) ** 2).mean())
                rc = resid.iloc[-126:] - resid.iloc[-126:].mean()
                crossings = int((np.sign(rc).diff().abs() == 2).sum())
                opps = int((z_val.abs() >= 1).astype(int).diff().eq(1).sum())
                v = vol.loc[p.index[-63] : p.index[-1], [t1, t2]].reindex(p.index[-63:])
                turnover1 = float((p[t1].iloc[-63:] * v[t1]).median())
                turnover2 = float((p[t2].iloc[-63:] * v[t2]).median())
                val_stats = static_validation(z_val, ret[t1].iloc[-126:], ret[t2].iloc[-126:])
                month_rows.append(
                    {
                        "screen_date": dt,
                        "ticker1": t1,
                        "ticker2": t2,
                        "pair": f"{t1}/{t2}",
                        "name1": UNIVERSE[t1][0],
                        "name2": UNIVERSE[t2][0],
                        "economic_link": economic_link(UNIVERSE[t1][1], UNIVERSE[t2][1]),
                        "corr252": corr,
                        "distance252": distance,
                        "eg_p252": eg_p,
                        "beta252": float(full_beta),
                        "half_life252": hl,
                        "beta_cv": bcv,
                        "crossings126": crossings,
                        "opportunities126": opps,
                        "turnover1_yen": turnover1,
                        "turnover2_yen": turnover2,
                        **val_stats,
                    }
                )
        if not month_rows:
            continue
        m = pd.DataFrame(month_rows)
        m["eg_q_bh"] = bh_qvalues(m["eg_p252"])
        # Fixed percent-rank score. No coefficient is estimated from future returns.
        higher = ["economic_link", "corr252", "crossings126", "opportunities126", "val_return", "val_sharpe"]
        lower = ["distance252", "eg_q_bh", "beta_cv"]
        rank_cols: dict[str, pd.Series] = {}
        for c in higher:
            rank_cols[c] = m[c].rank(pct=True, method="average")
        for c in lower:
            rank_cols[c] = 1 - m[c].rank(pct=True, method="average") + 1 / len(m)
        hl_score = np.exp(-np.abs(np.log(m["half_life252"].clip(lower=0.1) / 15.0)))
        m["score"] = (
            0.10 * rank_cols["economic_link"]
            + 0.15 * rank_cols["corr252"]
            + 0.08 * rank_cols["distance252"]
            + 0.12 * rank_cols["eg_q_bh"]
            + 0.08 * hl_score
            + 0.08 * rank_cols["beta_cv"]
            + 0.09 * rank_cols["crossings126"]
            + 0.08 * rank_cols["opportunities126"]
            + 0.10 * rank_cols["val_return"]
            + 0.12 * rank_cols["val_sharpe"]
        )
        m["eligible"] = (
            (m["turnover1_yen"] >= 5e8)
            & (m["turnover2_yen"] >= 5e8)
            & (m["corr252"] >= 0.35)
            & (m["beta252"] > 0)
            & (m["eg_q_bh"] <= 0.30)
            & m["half_life252"].between(2, 60)
            & (m["beta_cv"] <= 0.90)
            & (m["crossings126"] >= 3)
            & (m["opportunities126"] >= 4)
            & (m["val_trades"] >= 3)
        )
        # Nullable rank preserves pairs whose finite-score inputs are unavailable.
        m["overall_rank"] = m["score"].rank(ascending=False, method="min").round().astype("Int64")
        elig = m.loc[m["eligible"], "score"].rank(ascending=False, method="min")
        m["eligible_rank"] = np.nan
        m.loc[elig.index, "eligible_rank"] = elig
        m["shortlist"] = m["eligible"] & (m["eligible_rank"] <= 5)
        rows.extend(m.to_dict("records"))
    return pd.DataFrame(rows)


def strategy_matrix(prices: pd.DataFrame, model: pd.DataFrame, gates: dict[str, pd.Series], latest: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = {
        "design_2022_2024": (str(model.index.min().date()), "2024-12-30"),
        "validation_2025": ("2025-01-06", "2025-12-30"),
        "holdout_2026": ("2026-01-05", latest),
        "oos_2024_latest": ("2024-01-04", latest),
    }
    specs: dict[str, tuple[Params, str | None]] = {
        "low_always": (LOW, None),
        "low_structure_gate": (LOW, "structural"),
        "low_confirmed_gate": (LOW, "confirmed"),
        "conservative": (CONSERVATIVE, None),
    }
    for period, (start, end) in periods.items():
        for strategy, (params, gate_name) in specs.items():
            gate = gates.get(gate_name) if gate_name else None
            for tc in (5.0, 10.0, 20.0):
                for borrow in (0.01, 0.03, 0.05):
                    result = run_backtest(prices, model, params, start, end, tc, borrow, gate)
                    rows.append(
                        {
                            "period": period,
                            "start": start,
                            "end": end,
                            "strategy": strategy,
                            "tc_bp": tc,
                            "borrow_rate": borrow,
                            **result["metrics"],
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_trade_mean(trades: pd.DataFrame, reps: int = 5000, seed: int = 69816976) -> dict[str, float]:
    x = trades["trade_return"].to_numpy(dtype=float) if len(trades) else np.array([])
    if len(x) < 2:
        return {"trades": int(len(x)), "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_mean_le_zero": np.nan}
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return {
        "trades": int(len(x)),
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_mean_le_zero": float((means <= 0).mean()),
    }


def save_charts(
    prices: pd.DataFrame,
    model: pd.DataFrame,
    diagnostics: pd.DataFrame,
    strategy: pd.DataFrame,
    low_full: dict[str, object],
    pair_rank: pd.DataFrame,
    break_dates: list[pd.Timestamp],
) -> None:
    # 01: recent normalized prices, z, and low-threshold equity.
    d = low_full["daily"]
    assert isinstance(d, pd.DataFrame)
    recent_px = prices.loc["2025-01-01":, list(PAIR)].dropna()
    norm = recent_px / recent_px.iloc[0] * 100
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1]})
    axes[0].plot(norm.index, norm[PAIR[0]], color=BLUE, lw=1.6, label="村田製作所")
    axes[0].plot(norm.index, norm[PAIR[1]], color=ORANGE, lw=1.6, label="太陽誘電")
    axes[0].set_ylabel("指数 (2025年初=100)")
    axes[0].set_title("株価は同方向でも、相対価格は大きく往復")
    axes[0].legend(ncol=2, loc="upper left")
    axes[1].plot(model.loc["2025":].index, model.loc["2025":, "z"], color=NAVY, lw=1.1)
    for y, c, ls in [(1, RED, "--"), (-1, RED, "--"), (0.5, GREEN, ":"), (-0.5, GREEN, ":")]:
        axes[1].axhline(y, color=c, ls=ls, lw=0.9)
    axes[1].set_ylabel("z-score")
    axes[1].set_title("低閾値ルールの機会密度が上昇")
    axes[2].plot(d.loc["2025":].index, d.loc["2025":, "equity"] / d.loc["2025":, "equity"].iloc[0], color=PURPLE, lw=1.6)
    for bd in break_dates:
        axes[2].axvline(bd, color=GRAY, ls="--", lw=0.8, alpha=0.8)
    axes[2].set_ylabel("累積資産")
    axes[2].set_title("低閾値・常時稼働、純損益 (片道10bp + 借株年1%)")
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(CHARTS / "01_recent_regime.png", dpi=200)
    plt.close(fig)

    # 02: mechanism diagnostics and operational gates.
    fig, axes = plt.subplots(4, 1, figsize=(10.2, 8.0), sharex=True)
    sub = diagnostics.loc["2024":]
    axes[0].plot(sub.index, sub["corr126"], color=BLUE, label="126日相関")
    axes[0].axhline(0.45, color=GRAY, ls="--", lw=0.8)
    axes[0].set_ylabel("相関")
    axes[0].set_title("共通ファクター")
    axes[1].plot(sub.index, sub["half_life252"].clip(upper=80), color=ORANGE, label="半減期")
    axes[1].axhspan(2, 40, color=GREEN, alpha=0.08)
    axes[1].set_ylabel("営業日")
    axes[1].set_title("残差の収束速度")
    axes[2].plot(sub.index, sub["adf_p252"], color=NAVY, label="ADF p")
    axes[2].plot(sub.index, sub["eg_p252"], color=PURPLE, alpha=0.8, label="EG p")
    axes[2].axhline(0.20, color=GRAY, ls="--", lw=0.8)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("p値")
    axes[2].set_title("定常性は改善しても一貫しない")
    axes[2].legend(ncol=2, loc="upper right")
    axes[3].fill_between(sub.index, 0, sub["gate_structural"].astype(int), step="post", alpha=0.35, color=GREEN, label="構造ゲート")
    axes[3].fill_between(sub.index, 0, sub["gate_confirmed"].astype(int), step="post", alpha=0.45, color=PURPLE, label="実績確認ゲート")
    axes[3].set_ylim(-0.05, 1.15)
    axes[3].set_yticks([0, 1], ["OFF", "ON"])
    axes[3].set_title("当時入手可能な週次判定 (2週連続でON)")
    axes[3].legend(ncol=2, loc="upper left")
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(CHARTS / "02_diagnostics_gate.png", dpi=200)
    plt.close(fig)

    # 03: strategy PDCA holdout comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    base = strategy[(strategy.period == "holdout_2026") & (strategy.tc_bp == 10) & (strategy.borrow_rate == 0.01)]
    labels = {"low_always": "低閾値・常時", "low_structure_gate": "構造ゲート", "low_confirmed_gate": "確認ゲート", "conservative": "保守ルール"}
    x = np.arange(len(base))
    axes[0].bar(x, base["total_return"] * 100, color=[PURPLE, GREEN, BLUE, GRAY])
    axes[0].set_xticks(x, [labels[s] for s in base.strategy], rotation=18, ha="right")
    axes[0].axhline(0, color=GRAY, lw=0.8)
    axes[0].set_ylabel("累積収益 (%)")
    axes[0].set_title("2026年ホールドアウト")
    stress = strategy[(strategy.period == "holdout_2026") & (strategy.strategy == "low_always")]
    piv = stress.pivot(index="borrow_rate", columns="tc_bp", values="total_return") * 100
    im = axes[1].imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=min(-5, float(np.nanmin(piv.values))), vmax=max(5, float(np.nanmax(piv.values))))
    axes[1].set_xticks(range(len(piv.columns)), [f"{c:.0f}bp" for c in piv.columns])
    axes[1].set_yticks(range(len(piv.index)), [f"年{b:.0%}" for b in piv.index])
    axes[1].set_xlabel("片道取引コスト")
    axes[1].set_ylabel("借株料")
    axes[1].set_title("低閾値のコスト・借株ストレス")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[1].text(j, i, f"{piv.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], shrink=0.78, label="累積収益 (%)")
    fig.tight_layout()
    fig.savefig(CHARTS / "03_pdca_holdout.png", dpi=200)
    plt.close(fig)

    # 04: discovery rank and eligibility.
    fig, ax = plt.subplots(figsize=(10.2, 3.7))
    rank = pair_rank.set_index("screen_date")
    ax.plot(rank.index, rank["overall_rank"], color=BLUE, marker="o", ms=3, lw=1.3, label="全体順位")
    eligible_dates = rank.index[rank["eligible"]]
    ax.scatter(eligible_dates, rank.loc[eligible_dates, "overall_rank"], color=GREEN, s=28, zorder=3, label="適格")
    shortlist_dates = rank.index[rank["shortlist"]]
    ax.scatter(shortlist_dates, rank.loc[shortlist_dates, "overall_rank"], color=RED, s=36, marker="*", zorder=4, label="上位5候補")
    ax.invert_yaxis()
    ax.set_ylabel("順位 (低いほど上位)")
    ax.set_title("凍結ルールによる月次スクリーニング: 村田製作所 / 太陽誘電")
    ax.legend(ncol=3, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(CHARTS / "04_discovery_rank.png", dpi=200)
    plt.close(fig)


def main() -> None:
    prices, volume = fetch_universe()
    pair_prices = prices[list(PAIR)].dropna()
    actual_end = pair_prices.index.max()
    model = rolling_pair_model(pair_prices)
    model.to_csv(OUT / "pair_rolling_model.csv", encoding="utf-8-sig")

    # Always-on full history is used only as a lagged input to the detector.
    low_all = run_backtest(pair_prices, model, LOW, model.index.min(), actual_end)
    low_daily = low_all["daily"]
    assert isinstance(low_daily, pd.DataFrame)
    diagnostics = pair_diagnostics(pair_prices, model, low_daily)
    diagnostics.to_csv(OUT / "regime_diagnostics_weekly.csv", encoding="utf-8-sig")
    structural_gate = expand_weekly_gate(model.index, diagnostics["gate_structural"])
    confirmed_gate = expand_weekly_gate(model.index, diagnostics["gate_confirmed"])
    gates = {"structural": structural_gate, "confirmed": confirmed_gate}

    strategy = strategy_matrix(pair_prices, model, gates, str(actual_end.date()))
    strategy.to_csv(OUT / "strategy_cost_borrow_matrix.csv", index=False, encoding="utf-8-sig")

    # Exact 2026 base runs and daily/trades.
    runs: dict[str, dict[str, object]] = {
        "low_always": run_backtest(pair_prices, model, LOW, "2026-01-05", actual_end),
        "low_structure_gate": run_backtest(pair_prices, model, LOW, "2026-01-05", actual_end, gate=structural_gate),
        "low_confirmed_gate": run_backtest(pair_prices, model, LOW, "2026-01-05", actual_end, gate=confirmed_gate),
        "conservative": run_backtest(pair_prices, model, CONSERVATIVE, "2026-01-05", actual_end),
    }
    for name, result in runs.items():
        daily = result["daily"]
        trades = result["trades"]
        assert isinstance(daily, pd.DataFrame) and isinstance(trades, pd.DataFrame)
        daily.to_csv(OUT / f"2026_{name}_daily.csv", encoding="utf-8-sig")
        trades.to_csv(OUT / f"2026_{name}_trades.csv", index=False, encoding="utf-8-sig")

    # Offline ex-post weekly return change points; multiple minimum-segment choices give a date range.
    weekly_returns = low_daily.loc["2024":, "net_return"].resample("W-FRI").apply(lambda x: float((1 + x).prod() - 1))
    cp = [change_point_one_break(weekly_returns, m) for m in (8, 13, 21)]
    pd.DataFrame(cp).to_csv(OUT / "offline_change_points.csv", index=False, encoding="utf-8-sig")

    screen = screen_monthly(prices, volume)
    screen.to_csv(OUT / "monthly_pair_screen_all.csv", index=False, encoding="utf-8-sig")
    target_mask = ((screen.ticker1 == PAIR[0]) & (screen.ticker2 == PAIR[1])) | ((screen.ticker1 == PAIR[1]) & (screen.ticker2 == PAIR[0]))
    pair_rank = screen.loc[target_mask].sort_values("screen_date").copy()
    pair_rank.to_csv(OUT / "murata_taiyo_monthly_rank.csv", index=False, encoding="utf-8-sig")

    # Operational detector dates (activation transitions) and latest states.
    gate_info: dict[str, object] = {}
    for name, col in [("structural", "gate_structural"), ("confirmed", "gate_confirmed")]:
        s = diagnostics[col].fillna(False).astype(bool)
        previous = s.shift(1, fill_value=False).astype(bool)
        activations = s.index[s & ~previous]
        gate_info[name] = {
            "activations": [str(x.date()) for x in activations],
            "first_2026_activation": next((str(x.date()) for x in activations if x.year == 2026), None),
            "latest_state": bool(s.iloc[-1]),
            "latest_week": str(s.index[-1].date()),
        }

    # Sanity checks are assertions plus a machine-readable record.
    low2026 = runs["low_always"]
    d = low2026["daily"]
    tr = low2026["trades"]
    assert isinstance(d, pd.DataFrame) and isinstance(tr, pd.DataFrame)
    gross_exposure = d["w_murata"].abs() + d["w_taiyo"].abs()
    entries = int(((d["position_dir"] != 0) & (d["position_dir"].shift(1).fillna(0) == 0)).sum())
    pnl_error = float((d["gross_return"] - d["transaction_cost"] - d["borrow_cost"] - d["net_return"]).abs().max())
    screen_lookahead = bool((pd.to_datetime(screen["screen_date"]) <= actual_end).all())
    checks = {
        "data_cutoff_requested": str(AS_OF_REQUESTED.date()),
        "latest_common_pair_date": str(actual_end.date()),
        "latest_date_not_after_requested": bool(actual_end <= AS_OF_REQUESTED),
        "no_missing_pair_prices": bool(not pair_prices.isna().any().any()),
        "positive_pair_prices": bool((pair_prices > 0).all().all()),
        "max_abs_pnl_reconstruction_error": pnl_error,
        "gross_exposure_max": float(gross_exposure.max()),
        "gross_exposure_nonzero_min": float(gross_exposure[gross_exposure > 0].min()),
        "trade_count_equals_entries": bool(len(tr) == entries),
        "trade_count": int(len(tr)),
        "one_day_execution_lag": bool((d["position_dir"].iloc[1:-1].to_numpy() == d["target_dir"].shift(1).iloc[1:-1].fillna(0).to_numpy()).all()),
        "monthly_screen_uses_dates_through_cutoff": screen_lookahead,
        "screen_pair_count_latest": int((screen["screen_date"] == screen["screen_date"].max()).sum()),
        "screen_universe_size": len(UNIVERSE),
        "screen_survivorship_warning": "Universe is frozen using firms known at the 2026-09-02 research date; delisted/merged historical names are absent.",
    }
    assert checks["latest_date_not_after_requested"]
    assert checks["no_missing_pair_prices"]
    assert checks["positive_pair_prices"]
    assert pnl_error < 1e-12
    assert abs(checks["gross_exposure_max"] - 1.0) < 1e-12
    assert checks["trade_count_equals_entries"]
    assert checks["one_day_execution_lag"]
    with open(OUT / "validation_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    shortlist = pair_rank[pair_rank["shortlist"]]
    eligible = pair_rank[pair_rank["eligible"]]
    cp_dates = [pd.Timestamp(x["date"]) for x in cp if x.get("date")]
    latest_diag = diagnostics.iloc[-1]
    summary = {
        "requested_as_of": str(AS_OF_REQUESTED.date()),
        "actual_data_end": str(actual_end.date()),
        "rows_pair": int(len(pair_prices)),
        "model_first_date": str(model.index.min().date()),
        "universe_size": len(UNIVERSE),
        "candidate_pair_count": int(len(UNIVERSE) * (len(UNIVERSE) - 1) / 2),
        "low_params": asdict(LOW),
        "conservative_params": asdict(CONSERVATIVE),
        "holdout_2026": {k: v["metrics"] for k, v in runs.items()},
        "bootstrap_low_2026_trade_mean": bootstrap_trade_mean(runs["low_always"]["trades"]),
        "offline_change_points": cp,
        "offline_break_date_range": [min(str(x["date"]) for x in cp), max(str(x["date"]) for x in cp)],
        "gate_info": gate_info,
        "latest_diagnostics": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in latest_diag.items()},
        "pair_first_eligible_month": str(pd.Timestamp(eligible.iloc[0].screen_date).date()) if len(eligible) else None,
        "pair_first_shortlist_month": str(pd.Timestamp(shortlist.iloc[0].screen_date).date()) if len(shortlist) else None,
        "pair_latest_rank": int(pair_rank.iloc[-1].overall_rank) if len(pair_rank) else None,
        "pair_latest_eligible": bool(pair_rank.iloc[-1].eligible) if len(pair_rank) else None,
        "pair_latest_shortlist": bool(pair_rank.iloc[-1].shortlist) if len(pair_rank) else None,
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "yfinance": yf.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    with open(OUT / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_charts(pair_prices, model, diagnostics, strategy, low_all, pair_rank, cp_dates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

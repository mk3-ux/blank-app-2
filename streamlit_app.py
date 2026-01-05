from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from groq import Groq
except Exception:
    Groq = None


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Factor & Market Regime Analyzer (Educational)"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title=APP_NAME, layout="wide")

DISCLAIMER = """
⚠️ **Educational Use Only**

This application is for **financial education and learning only**.

• No investment advice  
• No recommendations  
• No buy / sell signals  
• Metrics are descriptive & historical  

Past performance does NOT guarantee future results.
"""


# ============================================================
# CORE FACTOR & REGIME ENGINE
# ============================================================

def analyze_factors_and_regime(portfolio: list[dict]) -> dict:
    if yf is None:
        raise RuntimeError("yfinance not available")

    tickers = [p["ticker"].upper() for p in portfolio]
    shares = {p["ticker"].upper(): p["shares"] for p in portfolio}

    prices = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="1y")
        if hist.empty:
            continue
        prices[t] = hist["Close"]

    prices_df = pd.DataFrame(prices).dropna()
    returns_df = prices_df.pct_change().dropna()

    latest_prices = prices_df.iloc[-1]

    # --------------------------------------------------------
    # Market values & weights
    # --------------------------------------------------------
    values = {
        t: latest_prices[t] * shares[t] for t in tickers
    }
    total_value = sum(values.values())
    weights = {t: v / total_value for t, v in values.items()}

    portfolio_returns = sum(
        returns_df[t] * weights[t] for t in tickers
    )

    # --------------------------------------------------------
    # Core portfolio metrics
    # --------------------------------------------------------
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative / cumulative.cummax()) - 1
    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Market comparison
    # --------------------------------------------------------
    spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
    aligned = portfolio_returns.align(spy, join="inner")

    beta = (
        np.cov(aligned[0], aligned[1])[0][1] / np.var(aligned[1])
        if len(aligned[0]) > 10 else 0
    )

    correlation = aligned[0].corr(aligned[1])

    # --------------------------------------------------------
    # Factor proxies (educational approximations)
    # --------------------------------------------------------
    momentum = returns_df.tail(60).mean().mean() * 252
    volatility_factor = returns_df.std().mean() * np.sqrt(252)

    concentration = max(weights.values())
    diversification_entropy = -sum(w * np.log(w) for w in weights.values())

    # --------------------------------------------------------
    # Market regime classification (educational)
    # --------------------------------------------------------
    if aligned[1].mean() > 0 and aligned[1].std() < 0.18:
        regime = "Risk-On (stable growth)"
    elif aligned[1].mean() < 0 and aligned[1].std() > 0.25:
        regime = "Risk-Off (stress / uncertainty)"
    else:
        regime = "Transitional / Mixed regime"

    # --------------------------------------------------------
    # Stock-level factor stats
    # --------------------------------------------------------
    stock_factors = {}
    for t in tickers:
        r = returns_df[t]
        stock_factors[t] = {
            "weight_pct": round(weights[t] * 100, 2),
            "annual_return": round(r.mean() * 252, 4),
            "annual_volatility": round(r.std() * np.sqrt(252), 4),
            "momentum_3m": round(r.tail(60).mean() * 252, 4),
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "portfolio_metrics": {
            "total_value": round(total_value, 2),
            "annual_return": round(annual_return, 4),
            "annual_volatility": round(annual_vol, 4),
            "max_drawdown": round(max_drawdown, 4),
            "beta_vs_market": round(beta, 2),
            "market_correlation": round(correlation, 2),
            "concentration_pct": round(concentration * 100, 2),
            "diversification_entropy": round(diversification_entropy, 3),
            "momentum_factor": round(momentum, 4),
            "volatility_factor": round(volatility_factor, 4),
            "market_regime": regime,
        },
        "stock_factors": stock_factors,
    }


def ai_explain_factors(result: dict) -> str | None:
    if not Groq or not GROQ_API_KEY:
        return None

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    Explain factor investing and market regimes in EDUCATIONAL terms only.

    No investment advice.
    No recommendations.

    Analysis summary:
    {result["portfolio_metrics"]}
    """

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Educational finance explainer only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
    )
    return resp.choices[0].message.content


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🧠 Factor & Market Regime Analyzer")
st.markdown(DISCLAIMER)
st.divider()

st.subheader("📥 Enter Portfolio (Educational)")
portfolio_input = st.text_area(
    "Format: TICKER,SHARES",
    value="AAPL,10\nMSFT,5\nVOO,3",
)

if st.button("Analyze Factors & Regime"):
    portfolio = []
    for line in portfolio_input.splitlines():
        try:
            t, s = line.split(",")
            portfolio.append({"ticker": t.strip(), "shares": float(s)})
        except Exception:
            pass

    if not portfolio:
        st.error("Invalid input format.")
    else:
        with st.spinner("Analyzing factors and regime..."):
            result = analyze_factors_and_regime(portfolio)

        st.success("Analysis complete")

        st.subheader("📊 Portfolio Factor Metrics")
        st.json(result["portfolio_metrics"])

        st.subheader("📄 Stock-Level Factor Exposure")
        st.dataframe(pd.DataFrame(result["stock_factors"]).T)

        explanation = ai_explain_factors(result)
        if explanation:
            st.subheader("🤖 AI Educational Explanation")
            st.markdown(explanation)

st.caption("Educational analysis only — no investment advice.")

"""Streamlit monitoring dashboard.

Run with:
    streamlit run src/trading_agent/dashboard.py

Shows:
- Live portfolio equity curve
- Open positions and P&L
- Recent trades table
- Signal history
- Scanner results
- Risk status indicators
- Alert log

Requires: pip install streamlit plotly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit dashboard."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        print("Dashboard requires streamlit and plotly:")
        print("  pip install streamlit plotly")
        sys.exit(1)

    from trading_agent.storage import TradeStore

    # ---- Page config ----
    st.set_page_config(
        page_title="Trading Agent Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 Trading Agent Dashboard")

    # ---- Connect to trade store ----
    db_path = st.sidebar.text_input("Database path", value="trading_history.db")
    if not Path(db_path).exists():
        st.warning(
            f"No database found at `{db_path}`. "
            "Run the trading agent first to generate data, or check the path."
        )
        st.stop()

    store = TradeStore(db_path)

    # ---- Sidebar ----
    st.sidebar.header("Filters")
    trade_limit = st.sidebar.slider("Trade history limit", 10, 500, 100)
    symbol_filter = st.sidebar.text_input("Filter by symbol", value="")

    # ---- Summary metrics ----
    summary = store.get_trade_summary()
    portfolio = store.get_portfolio_history(limit=1)
    latest_equity = portfolio[0]["equity"] if portfolio else 0.0
    latest_cash = portfolio[0]["cash"] if portfolio else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Equity", f"${latest_equity:,.2f}")
    col2.metric("Cash", f"${latest_cash:,.2f}")
    col3.metric("Total Trades", summary.get("total_trades", 0))
    col4.metric("Unique Symbols", summary.get("unique_symbols", 0))

    # ---- Equity curve ----
    st.subheader("Equity Curve")
    history = store.get_portfolio_history(limit=500)
    if history:
        history.reverse()
        timestamps = [h["timestamp"] for h in history]
        equities = [h["equity"] for h in history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=equities,
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=2),
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Time",
            yaxis_title="Equity ($)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No portfolio snapshots yet.")

    # ---- Recent trades ----
    st.subheader("Recent Trades")
    trades = store.get_trades(
        symbol=symbol_filter or None,
        limit=trade_limit,
    )
    if trades:
        import pandas as pd

        df = pd.DataFrame(trades)
        display_cols = ["timestamp", "symbol", "side", "qty", "price", "strategy", "status", "confidence"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet.")

    # ---- Signals ----
    with st.expander("Signal History"):
        signals = store.get_signals(limit=50)
        if signals:
            import pandas as pd

            df = pd.DataFrame(signals)
            display_cols = ["timestamp", "symbol", "signal_type", "confidence", "strategy"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No signals recorded yet.")

    # ---- Alerts ----
    st.subheader("Alerts")
    alerts = store.get_alerts(unacknowledged_only=False, limit=20)
    if alerts:
        for alert in alerts:
            level = alert.get("level", "info")
            icon = {"risk": "⚠️", "halt": "🛑", "error": "❌", "trade": "💰"}.get(level, "ℹ️")
            acked = "✅" if alert.get("acknowledged") else ""
            st.text(f"{icon} [{alert['timestamp']}] {alert['message']} {acked}")
    else:
        st.info("No alerts.")

    # ---- Daily P&L ----
    with st.expander("Daily P&L"):
        pnl = store.get_daily_pnl(days=30)
        if pnl:
            import pandas as pd

            df = pd.DataFrame(pnl)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for daily P&L.")

    # ---- Auto-refresh ----
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time

        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()

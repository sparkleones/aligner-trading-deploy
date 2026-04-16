"""
Aligner Trading — Live Dashboard (V14 Production)

Premium trading terminal — no external chart dependencies.
All data from Zerodha via engine. Everything visible at a glance.

Launch:
    streamlit run dashboard/live_dashboard.py --server.port 8505
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_bridge import DashboardStateReader


# ══════════════════════════════════════════════════════════════════════════
# ENGINE PROCESS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

ENGINE_PID_FILE = PROJECT_ROOT / "data" / "engine.pid"
ENGINE_SCRIPT = PROJECT_ROOT / "run_autonomous.py"
ENGINE_LOG_DIR = PROJECT_ROOT / "logs"


def _is_engine_running() -> bool:
    if not ENGINE_PID_FILE.exists():
        return False
    try:
        pid = int(ENGINE_PID_FILE.read_text().strip())
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x00100000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (ValueError, OSError, ProcessLookupError):
        return False


def _start_engine(paper: bool = False) -> tuple[bool, str]:
    if _is_engine_running():
        return False, "Engine is already running."
    ENGINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    log_file = ENGINE_LOG_DIR / f"engine_stdout_{date.today().isoformat()}.log"
    cmd = [sys.executable, str(ENGINE_SCRIPT)]
    if paper:
        cmd.append("--paper")
    try:
        with open(log_file, "a", encoding="utf-8") as lf:
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                )
            else:
                proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT), start_new_session=True,
                )
        ENGINE_PID_FILE.write_text(str(proc.pid))
        return True, f"Engine started (PID {proc.pid})"
    except Exception as e:
        return False, f"Failed to start engine: {e}"


def _stop_engine() -> tuple[bool, str]:
    if not ENGINE_PID_FILE.exists():
        return False, "No engine PID file found."
    try:
        pid = int(ENGINE_PID_FILE.read_text().strip())
        if sys.platform == "win32":
            os.kill(pid, 9)
        else:
            os.kill(pid, 15)
        time.sleep(1)
        ENGINE_PID_FILE.unlink(missing_ok=True)
        return True, f"Engine stopped (PID {pid})"
    except ProcessLookupError:
        ENGINE_PID_FILE.unlink(missing_ok=True)
        return True, "Engine was already stopped."
    except Exception as e:
        return False, f"Failed to stop engine: {e}"


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def pnl_color(val: float) -> str:
    return "#10b981" if val >= 0 else "#ef4444"


def pnl_bg(val: float) -> str:
    return "rgba(16,185,129,0.1)" if val >= 0 else "rgba(239,68,68,0.1)"


def fmt_hold(minutes: float) -> str:
    if not minutes or minutes < 0:
        return "0m"
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m" if h else f"{m}m"


def fmt_inr(val: float) -> str:
    """Format as INR — use HTML entity in markdown, plain text otherwise."""
    return f"Rs.{val:,.2f}"


def fmt_inr_html(val: float) -> str:
    """Format as INR with HTML rupee symbol for st.markdown."""
    return f"&#8377;{val:,.2f}"


# ══════════════════════════════════════════════════════════════════════════
# CHART BUILDERS (Plotly — Zerodha data only)
# ══════════════════════════════════════════════════════════════════════════

def build_candlestick_chart(
    bars: list[dict],
    support: float = 0,
    resistance: float = 0,
    open_positions: list[dict] | None = None,
    closed_positions: list[dict] | None = None,
    height: int = 520,
) -> go.Figure | None:
    """Candlestick + Volume chart from Zerodha bar data."""
    if not bars or len(bars) < 2:
        return None

    df = pd.DataFrame(bars)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    if df.empty:
        return None

    # Create subplot: candlestick on top, volume on bottom
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#10b981", increasing_fillcolor="#10b981",
        decreasing_line_color="#ef4444", decreasing_fillcolor="#ef4444",
        name="NIFTY", whiskerwidth=0.5,
    ), row=1, col=1)

    # ── VWAP ──
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, 1)
    vwap = (tp * vol).cumsum() / vol.cumsum()
    fig.add_trace(go.Scatter(
        x=df["time"], y=vwap, mode="lines", name="VWAP",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
    ), row=1, col=1)

    # ── EMA 20 ──
    if len(df) >= 20:
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df["time"], y=ema20, mode="lines", name="EMA 20",
            line=dict(color="#8b5cf6", width=1.2),
        ), row=1, col=1)

    # ── EMA 9 (fast) ──
    if len(df) >= 9:
        ema9 = df["close"].ewm(span=9, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df["time"], y=ema9, mode="lines", name="EMA 9",
            line=dict(color="#06b6d4", width=1),
        ), row=1, col=1)

    # ── Support / Resistance ──
    if support > 0:
        fig.add_hline(y=support, line_dash="dash", line_color="#10b981", opacity=0.5,
                      annotation_text=f"S: {support:,.0f}", annotation_position="bottom left",
                      annotation_font=dict(color="#10b981", size=10), row=1, col=1)
    if resistance > 0:
        fig.add_hline(y=resistance, line_dash="dash", line_color="#ef4444", opacity=0.5,
                      annotation_text=f"R: {resistance:,.0f}", annotation_position="top left",
                      annotation_font=dict(color="#ef4444", size=10), row=1, col=1)

    # ── Trade entry markers (open) ──
    if open_positions:
        for pos in open_positions:
            ep, et = pos.get("entry_price", 0), pos.get("entry_time", "")
            side = pos.get("side", "BUY")
            if ep > 0 and et:
                mc = "#3b82f6" if "BUY" in side.upper() else "#f97316"
                fig.add_trace(go.Scatter(
                    x=[et], y=[ep], mode="markers",
                    marker=dict(size=12, color=mc, symbol="triangle-up" if "BUY" in side.upper() else "triangle-down",
                                line=dict(width=1, color="white")),
                    name=f"Open: {pos.get('symbol', '')}", showlegend=True,
                    hovertext=f"{side} @ {ep:.1f}",
                ), row=1, col=1)

    # ── Closed trade markers ──
    if closed_positions:
        for pos in closed_positions:
            et = pos.get("exit_time", "")
            ex = pos.get("exit_price", 0)
            ppnl = pos.get("pnl", 0)
            if ex > 0 and et:
                mc = "#10b981" if ppnl >= 0 else "#ef4444"
                fig.add_trace(go.Scatter(
                    x=[et], y=[ex], mode="markers",
                    marker=dict(size=10, color=mc, symbol="x",
                                line=dict(width=2, color=mc)),
                    name=f"Exit {fmt_inr(ppnl)}", showlegend=False,
                    hovertext=f"Exit @ {ex:.1f} | P&L: {fmt_inr(ppnl)}",
                ), row=1, col=1)

    # ── Volume bars ──
    colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["time"], y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    # ── Layout ──
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e17",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=5, r=5, t=10, b=5),
        legend=dict(orientation="h", yanchor="top", y=1.06, xanchor="left", x=0,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        height=height, showlegend=True,
        xaxis2=dict(gridcolor="#1a2332"),
        yaxis=dict(gridcolor="#1a2332", showgrid=True, title=""),
        yaxis2=dict(gridcolor="#1a2332", showgrid=False, title="Vol"),
    )

    # Disable rangeslider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # X-axis: category type to avoid gaps, with time labels
    fig.update_xaxes(type="category", row=2, col=1)
    fig.update_xaxes(type="category", row=1, col=1)

    # Tick every Nth bar
    n_ticks = min(25, len(df))
    step = max(1, len(df) // n_ticks)
    tickvals = list(df["time"].iloc[::step])
    ticktext = [str(t).split(" ")[-1][:5] if " " in str(t) else str(t)[:5] for t in tickvals]
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    return fig


def build_pnl_chart(curve_data: list[dict], capital: float, kill_pct: float, height: int = 220) -> go.Figure | None:
    """Build P&L curve chart."""
    if not curve_data or len(curve_data) < 2:
        return None
    df = pd.DataFrame(curve_data)
    # Downsample to avoid rendering thousands of per-second points
    if len(df) > 200:
        step = len(df) // 200
        df = df.iloc[::step].reset_index(drop=True)
    last_pnl = df["pnl"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["pnl"], mode="lines", fill="tozeroy",
        line=dict(color=pnl_color(last_pnl), width=2),
        fillcolor=f"rgba({'16,185,129' if last_pnl >= 0 else '239,68,68'}, 0.1)",
        name="P&L",
    ))

    # Peak / trough annotations
    max_pnl = df["pnl"].max()
    min_pnl = df["pnl"].min()
    max_idx = df["pnl"].idxmax()
    min_idx = df["pnl"].idxmin()
    fig.add_trace(go.Scatter(
        x=[df.loc[max_idx, "time"]], y=[max_pnl], mode="markers",
        marker=dict(size=8, color="#10b981", symbol="diamond"),
        name=f"Peak: {fmt_inr(max_pnl)}", showlegend=True,
    ))
    if min_pnl < 0:
        fig.add_trace(go.Scatter(
            x=[df.loc[min_idx, "time"]], y=[min_pnl], mode="markers",
            marker=dict(size=8, color="#ef4444", symbol="diamond"),
            name=f"Trough: {fmt_inr(min_pnl)}", showlegend=True,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.3)")
    kill_level = -capital * kill_pct
    fig.add_hline(y=kill_level, line_dash="dot", line_color="#ef4444", opacity=0.4,
                  annotation_text=f"Kill Switch: {fmt_inr(kill_level)}",
                  annotation_position="bottom left",
                  annotation_font=dict(color="#ef4444", size=10))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e17",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=5, r=5, t=10, b=5),
        xaxis=dict(gridcolor="#1a2332", showgrid=True),
        yaxis=dict(gridcolor="#1a2332", showgrid=True, title="P&L"),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        height=height, showlegend=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Aligner Trading — V14",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS — Professional Trading Terminal Theme
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #0a0e17; --card: #111827; --card2: #151c2c; --border: #1e293b;
        --text: #e2e8f0; --muted: #64748b; --green: #10b981; --red: #ef4444;
        --blue: #3b82f6; --cyan: #06b6d4; --yellow: #f59e0b; --purple: #8b5cf6;
    }

    .stApp { background: var(--bg) !important; font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    [data-testid="stMetric"] {
        background: var(--card) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; padding: 10px 14px !important;
    }
    [data-testid="stMetric"] label {
        color: var(--muted) !important; font-size: 0.7rem !important;
        text-transform: uppercase !important; letter-spacing: 0.5px !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important;
        font-size: 1rem !important;
    }

    .card {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
    }
    .card-glass {
        background: rgba(17, 24, 39, 0.8); backdrop-filter: blur(12px);
        border: 1px solid rgba(100,116,139,0.15); border-radius: 10px;
        padding: 16px 20px; margin-bottom: 10px;
    }
    .hero-pnl {
        font-size: 2.4rem; font-weight: 800;
        font-family: 'JetBrains Mono', monospace; letter-spacing: -1px;
    }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 4px;
        font-size: 0.7rem; font-weight: 700; vertical-align: middle;
    }
    .mono { font-family: 'JetBrains Mono', monospace !important; }

    /* Position card hover */
    .pos-card {
        background: var(--card); border: 1px solid var(--border); border-radius: 8px;
        padding: 12px 16px; margin-bottom: 6px; transition: border-color 0.2s;
    }
    .pos-card:hover { border-color: var(--cyan); }

    /* Order status badges */
    .status-filled { color: #10b981; background: rgba(16,185,129,0.15); padding: 2px 8px; border-radius: 3px; font-size: 0.7rem; }
    .status-rejected { color: #ef4444; background: rgba(239,68,68,0.15); padding: 2px 8px; border-radius: 3px; font-size: 0.7rem; }
    .status-pending { color: #f59e0b; background: rgba(245,158,11,0.15); padding: 2px 8px; border-radius: 3px; font-size: 0.7rem; }

    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background: var(--card); border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; border-radius: 6px !important;
        color: var(--muted) !important; padding: 8px 20px !important;
        font-weight: 600 !important; font-size: 0.8rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--border) !important; color: var(--text) !important;
    }

    /* Button styling */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-weight: 700 !important; border-radius: 6px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
    @keyframes glow { 0%,100%{box-shadow:0 0 5px rgba(16,185,129,0.3);} 50%{box-shadow:0 0 15px rgba(16,185,129,0.6);} }

    /* Prevent white flash on refresh — set background at HTML/body level */
    html, body {
        background-color: #0a0e17 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    [data-testid="stHeader"] {
        background-color: #0a0e17 !important;
        transition: none !important;
    }
    /* Hide autorefresh iframe */
    iframe[title="streamlit_autorefresh.st_autorefresh"] {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        border: none !important;
        overflow: hidden !important;
    }
    /* Stable chart containers — prevent layout shift */
    .stPlotlyChart {
        min-height: 200px;
    }
    /* Prevent element flickering during rerun */
    .stMarkdown, [data-testid="stMetric"], .stTabs, .stExpander {
        animation: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TOP CONTROL BAR
# ══════════════════════════════════════════════════════════════════════════

engine_running = _is_engine_running()

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5, ctrl_c6 = st.columns([1.5, 1, 1, 1.5, 1.2, 1.8])

with ctrl_c1:
    if engine_running:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                         background:#10b981;animation:pulse 2s infinite;"></span>
            <span style="color:#10b981;font-weight:700;font-size:0.9rem;">ENGINE RUNNING</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;"></span>
            <span style="color:#ef4444;font-weight:700;font-size:0.9rem;">ENGINE STOPPED</span>
        </div>""", unsafe_allow_html=True)

with ctrl_c2:
    if not engine_running:
        if st.button("Start LIVE", type="primary", use_container_width=True):
            ok, msg = _start_engine(paper=False)
            st.toast(msg); time.sleep(2); st.rerun()
    else:
        if st.button("Stop Engine", type="secondary", use_container_width=True):
            ok, msg = _stop_engine()
            st.toast(msg); time.sleep(1); st.rerun()

with ctrl_c3:
    if not engine_running:
        if st.button("Start PAPER", use_container_width=True):
            ok, msg = _start_engine(paper=True)
            st.toast(msg); time.sleep(2); st.rerun()

with ctrl_c4:
    st.markdown("""
    <div style="padding:6px 0;">
        <span style="background:linear-gradient(135deg,#3b82f6,#06b6d4);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     font-weight:900;font-size:1.1rem;">V14 PRODUCTION</span>
        <span style="color:var(--muted);font-size:0.65rem;margin-left:6px;">VWAP+RSI+Squeeze</span>
    </div>""", unsafe_allow_html=True)

with ctrl_c5:
    auto_refresh = st.toggle("Auto Refresh", value=True, key="auto_refresh_main")

with ctrl_c6:
    refresh_speed = st.select_slider(
        "Speed", options=["1s", "2s", "3s", "5s"],
        value="1s", key="refresh_speed_main", label_visibility="collapsed",
    )

refresh_ms = max(1000, int(float(refresh_speed.replace("s", "")) * 1000))
if auto_refresh:
    st_autorefresh(interval=refresh_ms, limit=None, key="refresh")


# ══════════════════════════════════════════════════════════════════════════
# LOAD STATE
# ══════════════════════════════════════════════════════════════════════════

state = DashboardStateReader.read_state()

if not state:
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:60px 20px;">
        <div style="font-size:2.2rem;font-weight:800;margin-bottom:12px;">
            <span style="background:linear-gradient(135deg,#3b82f6,#06b6d4);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                ALIGNER TRADING
            </span>
        </div>
        <div style="font-size:0.95rem;color:var(--muted);margin-bottom:20px;">
            V14 Production &bull; VWAP + RSI + Squeeze Confluence Engine
        </div>
        <div style="font-size:1.1rem;color:var(--text);">
            {'Engine started — waiting for first data...' if engine_running else 'Click <b>Start LIVE</b> or <b>Start PAPER</b> to begin.'}
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Parse state ──
sys_state = state.get("system", {})
market = state.get("market", {})
positions = state.get("positions", {})
pnl_data = state.get("pnl", {})
risk = state.get("risk", {})
orders = state.get("orders", [])
agent_info = state.get("agent", {})

status_val = sys_state.get("status", "UNKNOWN")
mode_val = sys_state.get("mode", "")
capital = sys_state.get("capital", 0)
spot = market.get("spot_price", 0)
prev_close = market.get("prev_close", 0)
day_change = market.get("day_change", 0)
day_change_pct = market.get("day_change_pct", 0)
total_pnl = pnl_data.get("total", 0)
total_pnl_pct = pnl_data.get("total_pct", 0)
realized = pnl_data.get("realized", 0)
unrealized = pnl_data.get("unrealized", 0)
gross_pnl = pnl_data.get("gross", realized + unrealized)
est_charges = pnl_data.get("charges", 0)
vix_val = market.get("vix", 0)
pcr_val = market.get("pcr", 0)
bias_val = market.get("market_bias", "")
confidence_val = market.get("confidence", 0)
support_val = market.get("support", 0)
resistance_val = market.get("resistance", 0)
is_expiry = market.get("is_expiry_day", False)
bars_done = sys_state.get("bars_processed", 0)
total_bars_day = sys_state.get("total_bars", 375)
agent_name = agent_info.get("name", "v14_production")
signals_gen = agent_info.get("signals_generated", 0)
signals_acc = agent_info.get("signals_accepted", 0)
signals_filt = agent_info.get("signals_filtered", 0)
last_signal = agent_info.get("last_signal", "")
trades_today = risk.get("trades_today", 0)
winners = risk.get("winners", 0)
losers = risk.get("losers", 0)
kill_triggered = risk.get("kill_switch_triggered", False)
kill_pct = risk.get("kill_switch_pct", 0.03)
daily_loss_pct = risk.get("daily_loss_pct", 0)
open_count = risk.get("open_count", 0)
max_pos = risk.get("max_positions", 4)
max_trades = risk.get("max_trades_per_day", 5)
wr = round(winners / (winners + losers) * 100) if (winners + losers) > 0 else 0

open_pos = positions.get("open", [])
closed_pos = positions.get("closed", [])
btst_pos = positions.get("btst", [])

# Staleness
try:
    updated_dt = datetime.fromisoformat(state.get("last_updated", ""))
    staleness = (datetime.now() - updated_dt).total_seconds()
    is_stale = staleness > 30
except Exception:
    staleness = 999; is_stale = True

now = datetime.now()
mkt_close = now.replace(hour=15, minute=30, second=0)
time_display = f"{(mkt_close - now).seconds // 3600}h {((mkt_close - now).seconds % 3600) // 60}m to close" if now < mkt_close else "Market Closed"


# ══════════════════════════════════════════════════════════════════════════
# HEADER — Hero section with key numbers
# ══════════════════════════════════════════════════════════════════════════

status_color = "#10b981" if status_val == "TRADING" else "#f59e0b" if status_val != "STOPPED" else "#ef4444"
dot = "background:#ef4444" if is_stale else "background:#10b981;animation:pulse 2s infinite"
expiry_badge = '<span class="badge" style="background:#f59e0b;color:#000;margin-left:8px;">EXPIRY DAY</span>' if is_expiry else ""

st.markdown(f"""
<div class="card-glass">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div>
            <div style="font-size:1.5rem;font-weight:800;color:var(--text);">
                <span style="display:inline-block;width:9px;height:9px;border-radius:50%;{dot};margin-right:8px;"></span>
                NIFTY 50
                <span class="badge" style="border:1px solid {status_color};color:{status_color};margin-left:10px;">{status_val} {mode_val}</span>
                <span class="badge" style="border:1px solid #06b6d4;color:#06b6d4;margin-left:4px;">{agent_name}</span>
                {expiry_badge}
            </div>
            <div style="color:var(--muted);font-size:0.9rem;margin-top:6px;">
                <span class="mono" style="color:var(--text);font-weight:700;font-size:1.15rem;">{spot:,.2f}</span>
                <span class="mono" style="color:{pnl_color(day_change)};margin-left:8px;">
                    {day_change:+.2f} ({day_change_pct:+.2f}%)
                </span>
                &nbsp;&bull;&nbsp; Capital: <span class="mono">{fmt_inr_html(capital)}</span>
                &nbsp;&bull;&nbsp; {time_display}
            </div>
        </div>
        <div style="text-align:right;">
            <div class="hero-pnl" style="color:{pnl_color(total_pnl)};">
                {fmt_inr_html(total_pnl)}
            </div>
            <div style="color:{pnl_color(total_pnl)};font-size:0.9rem;font-weight:600;">
                {total_pnl_pct:+.2f}% &bull; {trades_today} trades &bull; {wr}% win rate
            </div>
            {f'<div style="color:var(--muted);font-size:0.68rem;margin-top:2px;">Charges: &#8377;{est_charges:,.0f} &bull; Gross: &#8377;{gross_pnl:+,.0f}</div>' if est_charges > 0 else ''}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Metrics Strip ──
m1, m2, m3, m4, m5, m6, m7, m8, m9 = st.columns(9)
with m1: st.metric("VIX", f"{vix_val:.1f}")
with m2: st.metric("PCR", f"{pcr_val:.2f}", bias_val)
with m3: st.metric("Support", f"{support_val:,.0f}")
with m4: st.metric("Resistance", f"{resistance_val:,.0f}")
with m5: st.metric("Realized", f"{realized:+,.0f}")
with m6: st.metric("Unrealized", f"{unrealized:+,.0f}")
with m7: st.metric("Charges", f"-{est_charges:,.0f}" if est_charges > 0 else "0")
with m8: st.metric("Signals", f"{signals_acc}/{signals_gen}", f"{signals_filt} filtered")
with m9: st.metric("Bars", f"{bars_done}/{total_bars_day}", f"Conf: {confidence_val:.0%}")


# ══════════════════════════════════════════════════════════════════════════
# LIVE INDICES TICKER
# ══════════════════════════════════════════════════════════════════════════

indices = state.get("indices", [])
if indices:
    ticker_parts = []
    for idx in indices:
        name = idx.get("name", "")
        ltp = idx.get("ltp", 0)
        chg = idx.get("change", 0)
        chg_pct = idx.get("change_pct", 0)
        c = "#10b981" if chg >= 0 else "#ef4444"
        arrow = "&#9650;" if chg >= 0 else "&#9660;"
        ticker_parts.append(
            f'<div style="flex:0 0 auto;padding:6px 16px;border-right:1px solid var(--border);">'
            f'<span style="color:var(--muted);font-size:0.7rem;font-weight:600;">{name}</span>'
            f'<br/><span class="mono" style="color:var(--text);font-size:0.85rem;font-weight:700;">'
            f'{ltp:,.2f}</span> '
            f'<span class="mono" style="color:{c};font-size:0.72rem;">{arrow} {chg_pct:+.2f}%</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;overflow-x:auto;gap:0;background:var(--card);'
        f'border:1px solid var(--border);border-radius:8px;margin-bottom:10px;">'
        + "".join(ticker_parts)
        + '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# CLAUDE AI BRAIN — Market Intelligence
# ══════════════════════════════════════════════════════════════════════════

claude_data = DashboardStateReader.read_claude_brain()
claude_analysis = claude_data.get("analysis", {})
claude_enabled = claude_data.get("enabled", False)

if claude_analysis and claude_analysis.get("one_liner"):
    ca = claude_analysis
    regime = ca.get("regime", "unknown")
    conviction = ca.get("conviction", "low")
    rec_action = ca.get("recommended_action", "HOLD")
    risk_level = ca.get("risk_assessment", "moderate")
    one_liner = ca.get("one_liner", "")
    detailed = ca.get("detailed_analysis", "")
    override = ca.get("override_confluence", False)
    override_reason = ca.get("override_reason", "")
    pos_advice = ca.get("position_advice", "")
    key_levels = ca.get("key_levels", {})

    # Color by conviction
    conv_colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#64748b"}
    conv_color = conv_colors.get(conviction, "#64748b")
    action_colors = {"BUY_CALL": "#10b981", "BUY_PUT": "#ef4444", "HOLD": "#64748b",
                     "EXIT_ALL": "#ef4444", "REDUCE_SIZE": "#f59e0b"}
    action_color = action_colors.get(rec_action, "#64748b")

    override_html = ""
    if override:
        override_html = (
            '<div style="margin-top:6px;padding:6px 10px;background:rgba(16,185,129,0.15);'
            'border:1px solid rgba(16,185,129,0.3);border-radius:6px;font-size:0.72rem;">'
            '<span style="color:#10b981;font-weight:700;">CONFLUENCE OVERRIDE:</span> '
            f'<span style="color:var(--text);">{override_reason}</span>'
            '</div>'
        )

    st.markdown(f"""
    <div class="card-glass" style="border-left:3px solid {conv_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div>
                <span style="color:#8b5cf6;font-weight:800;font-size:0.8rem;letter-spacing:1px;">CLAUDE AI</span>
                <span class="badge" style="background:{action_color};color:white;margin-left:8px;">{rec_action}</span>
                <span class="badge" style="border:1px solid {conv_color};color:{conv_color};margin-left:4px;">{conviction.upper()}</span>
                <span style="color:var(--muted);font-size:0.72rem;margin-left:8px;">
                    Regime: {regime} | Risk: {risk_level}
                </span>
            </div>
            <span style="color:var(--muted);font-size:0.65rem;">
                {ca.get('timestamp','')[:19]}
            </span>
        </div>
        <div style="color:var(--text);font-size:0.85rem;margin-top:6px;font-weight:500;">
            {one_liner}
        </div>
        <div style="color:var(--muted);font-size:0.72rem;margin-top:4px;">
            {detailed}
        </div>
        {f'<div style="color:var(--muted);font-size:0.72rem;margin-top:4px;">Positions: {pos_advice}</div>' if pos_advice else ''}
        {override_html}
    </div>
    """, unsafe_allow_html=True)
elif not claude_enabled:
    st.markdown("""
    <div class="card-glass" style="border-left:3px solid #64748b;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="color:#8b5cf6;font-weight:800;font-size:0.8rem;letter-spacing:1px;">CLAUDE AI</span>
            <span style="color:var(--muted);font-size:0.78rem;">
                Add ANTHROPIC_API_KEY to .env to enable AI market analysis
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# V14 DECISION PANEL — What the model is waiting for
# ══════════════════════════════════════════════════════════════════════════

dec = state.get("decision", {})
if dec.get("ready"):
    put_sc = dec.get("put_score", 0)
    put_min = dec.get("put_min", 4)
    call_sc = dec.get("call_score", 0)
    call_min = dec.get("call_min", 5)
    d_rsi = dec.get("rsi", 50)
    d_vwap = dec.get("vwap", 0)
    d_spot = dec.get("spot", 0)
    d_st = dec.get("supertrend", "?")
    d_ema = dec.get("ema9_above_ema21", False)
    d_sq = dec.get("squeeze", False)

    put_pct = min(100, put_sc / put_min * 100) if put_min > 0 else 0
    call_pct = min(100, call_sc / call_min * 100) if call_min > 0 else 0
    put_bar_color = "#ef4444" if put_pct >= 100 else "#ef4444" if put_pct >= 70 else "#64748b"
    call_bar_color = "#10b981" if call_pct >= 100 else "#10b981" if call_pct >= 70 else "#64748b"

    # PUT triggers
    put_trig_html = ""
    for t in dec.get("put_triggers", []):
        put_trig_html += f'<div style="color:#ef4444;font-size:0.72rem;">&#9660; {t}</div>'
    if not put_trig_html and dec.get("put_ready"):
        put_trig_html = '<div style="color:#ef4444;font-size:0.72rem;font-weight:700;">READY TO FIRE</div>'

    # CALL triggers
    call_trig_html = ""
    for t in dec.get("call_triggers", []):
        call_trig_html += f'<div style="color:#10b981;font-size:0.72rem;">&#9650; {t}</div>'
    if not call_trig_html and dec.get("call_ready"):
        call_trig_html = '<div style="color:#10b981;font-size:0.72rem;font-weight:700;">READY TO FIRE</div>'

    # Score breakdowns
    put_bd = " + ".join(dec.get("put_breakdown", []))
    call_bd = " + ".join(dec.get("call_breakdown", []))

    spot_vs_vwap = "ABOVE" if d_spot > d_vwap else "BELOW"
    svw_color = "#10b981" if d_spot > d_vwap else "#ef4444"

    st.markdown(f"""
    <div class="card-glass">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="color:#f59e0b;font-weight:800;font-size:0.8rem;letter-spacing:1px;">V14 DECISION PANEL</span>
            <span style="color:var(--muted);font-size:0.7rem;">
                ST: {d_st} | EMA: {'9>21' if d_ema else '9<21'} | RSI: {d_rsi:.0f}
                | Squeeze: {'ON' if d_sq else 'OFF'}
                | Spot <span style="color:{svw_color}">{spot_vs_vwap}</span> VWAP ({d_vwap:,.0f})
            </span>
        </div>
        <div style="display:flex;gap:16px;">
            <div style="flex:1;background:var(--card);border-radius:8px;padding:12px;border:1px solid var(--border);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="color:#ef4444;font-weight:700;font-size:0.85rem;">BUY PUT</span>
                    <span class="mono" style="color:#ef4444;font-size:0.85rem;">{put_sc:.1f}/{put_min:.0f}</span>
                </div>
                <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden;margin-bottom:8px;">
                    <div style="background:{put_bar_color};height:100%;width:{put_pct:.0f}%;border-radius:4px;transition:width 0.3s;"></div>
                </div>
                <div style="color:var(--muted);font-size:0.65rem;margin-bottom:6px;">{put_bd}</div>
                <div style="border-top:1px solid var(--border);padding-top:6px;margin-top:4px;">
                    <div style="color:var(--muted);font-size:0.68rem;font-weight:600;margin-bottom:3px;">Waiting for:</div>
                    {put_trig_html or '<div style="color:var(--muted);font-size:0.72rem;">All conditions met</div>'}
                </div>
            </div>
            <div style="flex:1;background:var(--card);border-radius:8px;padding:12px;border:1px solid var(--border);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="color:#10b981;font-weight:700;font-size:0.85rem;">BUY CALL</span>
                    <span class="mono" style="color:#10b981;font-size:0.85rem;">{call_sc:.1f}/{call_min:.0f}</span>
                </div>
                <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden;margin-bottom:8px;">
                    <div style="background:{call_bar_color};height:100%;width:{call_pct:.0f}%;border-radius:4px;transition:width 0.3s;"></div>
                </div>
                <div style="color:var(--muted);font-size:0.65rem;margin-bottom:6px;">{call_bd}</div>
                <div style="border-top:1px solid var(--border);padding-top:6px;margin-top:4px;">
                    <div style="color:var(--muted);font-size:0.68rem;font-weight:600;margin-bottom:3px;">Waiting for:</div>
                    {call_trig_html or '<div style="color:var(--muted);font-size:0.72rem;">All conditions met</div>'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif dec.get("reason"):
    st.markdown(f"""
    <div class="card-glass">
        <span style="color:#f59e0b;font-weight:800;font-size:0.8rem;letter-spacing:1px;">V14 DECISION PANEL</span>
        <span style="color:var(--muted);font-size:0.78rem;margin-left:12px;">{dec['reason']}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════

chart_col, right_col = st.columns([3, 2])


# ─────────────────── LEFT: Charts ───────────────────────────────────────

with chart_col:

    # ── NIFTY Candlestick Chart ──
    bars_data = market.get("bars", [])

    nifty_fig = build_candlestick_chart(
        bars=bars_data, support=support_val, resistance=resistance_val,
        open_positions=open_pos, closed_positions=closed_pos, height=500,
    )

    if nifty_fig:
        st.plotly_chart(nifty_fig, key="nifty_chart",
                        config={"displayModeBar": True, "scrollZoom": True})
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:80px 20px;color:var(--muted);">
            <div style="font-size:1.4rem;margin-bottom:8px;">NIFTY Chart</div>
            <div style="font-size:0.85rem;">Candlestick chart appears once the engine streams bar data from Zerodha.</div>
            <div style="font-size:0.8rem;margin-top:8px;color:var(--cyan);">
                VWAP &bull; EMA 9/20 &bull; Support/Resistance &bull; Trade Markers
            </div>
        </div>""", unsafe_allow_html=True)

    # ── P&L Curve ──
    curve_data = pnl_data.get("curve", [])
    pnl_fig = build_pnl_chart(curve_data, capital, kill_pct, height=200)
    if pnl_fig:
        st.plotly_chart(pnl_fig, key="pnl_curve")


# ─────────────────── RIGHT: Positions + Trades + Orders ─────────────────

with right_col:

    pos_tab, history_tab, orders_tab = st.tabs(["Positions", "Trade History", "Orders"])

    # ── TAB 1: Open Positions ──
    with pos_tab:
        if open_pos:
            for pos in open_pos:
                sym = pos.get("symbol", "")
                side = pos.get("side", "")
                qty = pos.get("qty", 0)
                entry = pos.get("entry_price", 0)
                current = pos.get("current_price", 0)
                pos_pnl = pos.get("pnl", 0)
                strategy = pos.get("strategy", "")
                entry_time = pos.get("entry_time", "")
                hold_mins = pos.get("hold_minutes", 0)
                pnl_pct = (pos_pnl / (entry * qty) * 100) if entry and qty else 0

                side_color = "#3b82f6" if "BUY" in side.upper() else "#f97316"

                st.markdown(f"""
                <div class="pos-card" style="border-left:3px solid {pnl_color(pos_pnl)};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:var(--text);font-weight:700;font-size:0.95rem;">{sym}</span>
                            <span class="badge" style="background:{side_color};color:white;margin-left:8px;">{side}</span>
                            <span style="color:var(--muted);font-size:0.75rem;margin-left:4px;">x{qty}</span>
                        </div>
                        <div style="text-align:right;">
                            <span class="mono" style="color:{pnl_color(pos_pnl)};font-weight:800;font-size:1.15rem;">
                                {fmt_inr_html(pos_pnl)}
                            </span>
                            <span class="mono" style="color:{pnl_color(pos_pnl)};font-size:0.75rem;margin-left:6px;">
                                ({pnl_pct:+.1f}%)
                            </span>
                        </div>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:var(--muted);font-size:0.72rem;margin-top:6px;">
                        <span>Entry: <span class="mono">{entry:.2f}</span> @ {entry_time}</span>
                        <span>LTP: <span class="mono" style="color:var(--text);">{current:.2f}</span></span>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:var(--muted);font-size:0.72rem;margin-top:3px;">
                        <span>Hold: {fmt_hold(hold_mins)}</span>
                        <span>{strategy}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;color:var(--muted);padding:40px;">
                No open positions
            </div>""", unsafe_allow_html=True)

        # ── Summary stats ──
        if open_pos:
            total_unrealized = sum(p.get("pnl", 0) for p in open_pos)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 12px;background:var(--card);
                        border-radius:6px;margin-top:8px;font-size:0.78rem;">
                <span style="color:var(--muted);">Open P&L:</span>
                <span class="mono" style="color:{pnl_color(total_unrealized)};font-weight:700;">
                    {fmt_inr_html(total_unrealized)}
                </span>
            </div>""", unsafe_allow_html=True)

        # BTST Positions
        if btst_pos:
            st.markdown("---")
            st.markdown(f"**BTST Positions ({len(btst_pos)})**")
            for bp in btst_pos:
                bsym = bp.get("symbol", "")
                bpnl = bp.get("pnl", 0)
                st.markdown(f"""
                <div class="pos-card" style="border-left:3px solid #f59e0b;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:var(--text);font-weight:600;">{bsym}</span>
                        <span class="mono" style="color:{pnl_color(bpnl)}">{fmt_inr_html(bpnl)}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 2: Closed Trades / Trade History ──
    with history_tab:
        if closed_pos:
            # Summary row
            total_realized = sum(p.get("pnl", 0) for p in closed_pos)
            avg_hold = sum(p.get("hold_minutes", 0) for p in closed_pos) / len(closed_pos)
            best_trade = max(closed_pos, key=lambda x: x.get("pnl", 0))
            worst_trade = min(closed_pos, key=lambda x: x.get("pnl", 0))

            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("Total P&L", f"{total_realized:+,.0f}")
            with s2: st.metric("Trades", f"{len(closed_pos)}")
            with s3: st.metric("Best", f"{best_trade.get('pnl',0):+,.0f}")
            with s4: st.metric("Worst", f"{worst_trade.get('pnl',0):+,.0f}")

            # Each closed trade as a compact card
            for ct in reversed(closed_pos):
                ct_sym = ct.get("symbol", "")
                ct_side = ct.get("side", "")
                ct_pnl = ct.get("pnl", 0)
                ct_entry = ct.get("entry_price", 0)
                ct_exit = ct.get("exit_price", 0)
                ct_entry_t = ct.get("entry_time", "")
                ct_exit_t = ct.get("exit_time", "")
                ct_hold = ct.get("hold_minutes", 0)
                ct_type = ct.get("entry_type", "")
                ct_reason = ct.get("exit_reason", "")

                st.markdown(f"""
                <div class="pos-card" style="border-left:3px solid {pnl_color(ct_pnl)};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:var(--text);font-weight:600;font-size:0.85rem;">{ct_sym}</span>
                            <span style="color:var(--muted);font-size:0.7rem;margin-left:6px;">{ct_side}</span>
                        </div>
                        <span class="mono" style="color:{pnl_color(ct_pnl)};font-weight:800;">
                            {fmt_inr_html(ct_pnl)}
                        </span>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:var(--muted);font-size:0.68rem;margin-top:4px;">
                        <span>In: <span class="mono">{ct_entry:.2f}</span> &rarr; Out: <span class="mono">{ct_exit:.2f}</span></span>
                        <span>{fmt_hold(ct_hold)} &bull; {ct_reason or ct_type}</span>
                    </div>
                    <div style="color:var(--muted);font-size:0.65rem;margin-top:2px;">
                        {ct_entry_t} &rarr; {ct_exit_t}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;color:var(--muted);padding:40px;">
                No trades closed today
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: Orders ──
    with orders_tab:
        all_orders = orders
        if not all_orders:
            order_log = DashboardStateReader.read_order_log()
            if order_log:
                today_str = date.today().isoformat()
                all_orders = [o for o in order_log if o.get("placed_at", "").startswith(today_str)]

        if all_orders:
            filled = sum(1 for o in all_orders if o.get("status") in ("COMPLETE", "FILLED", "EXECUTED", "TRADED"))
            failed = sum(1 for o in all_orders if o.get("status") in ("REJECTED", "CANCELLED", "FAILED"))
            pending = len(all_orders) - filled - failed

            o1, o2, o3, o4 = st.columns(4)
            with o1: st.metric("Total", f"{len(all_orders)}")
            with o2: st.metric("Filled", f"{filled}")
            with o3: st.metric("Failed", f"{failed}")
            with o4: st.metric("Pending", f"{pending}")

            for order in reversed(all_orders):
                o_sym = order.get("symbol", "")
                o_side = order.get("side", "")
                o_qty = order.get("qty", 0)
                o_status = order.get("status", "")
                o_fill = order.get("fill_price", 0)
                o_time = order.get("placed_at", "")
                o_error = order.get("error", "")
                o_tag = order.get("tag", "")

                if o_status in ("COMPLETE", "FILLED", "EXECUTED", "TRADED"):
                    status_cls = "status-filled"
                elif o_status in ("REJECTED", "CANCELLED", "FAILED"):
                    status_cls = "status-rejected"
                else:
                    status_cls = "status-pending"

                st.markdown(f"""
                <div class="pos-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:var(--text);font-weight:600;font-size:0.85rem;">{o_sym}</span>
                            <span style="color:var(--muted);font-size:0.7rem;margin-left:6px;">{o_side} x{o_qty}</span>
                        </div>
                        <span class="{status_cls}">{o_status}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;color:var(--muted);font-size:0.68rem;margin-top:4px;">
                        <span>{o_time}</span>
                        <span>{'Fill: <span class=mono>' + f'{o_fill:.2f}</span>' if o_fill else ''}{' | ' + o_tag if o_tag else ''}</span>
                    </div>
                    {'<div style="color:#ef4444;font-size:0.65rem;margin-top:2px;">' + o_error + '</div>' if o_error else ''}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;color:var(--muted);padding:40px;">
                No orders placed today
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MANUAL ORDER PANEL
# ══════════════════════════════════════════════════════════════════════════

with st.expander("Manual Order", expanded=engine_running):
    import uuid

    # Strike selection based on current spot
    strike_interval = 50
    atm_strike = round(spot / strike_interval) * strike_interval if spot > 0 else 23000
    strikes = [atm_strike + i * strike_interval for i in range(-10, 11)]

    # Expiry: next Tuesday (NIFTY weekly)
    _today = date.today()
    _days_to_tue = (1 - _today.weekday()) % 7
    if _days_to_tue == 0:
        _expiry = _today  # Today is Tuesday
    else:
        _expiry = _today + timedelta(days=_days_to_tue)
    _expiry_str = _expiry.strftime("%y%m%d")

    mo_c1, mo_c2, mo_c3, mo_c4 = st.columns([2.5, 1, 1, 1])
    with mo_c1:
        selected_strike = st.selectbox(
            "Strike", strikes, index=10, key="mo_strike",
            format_func=lambda x: f"{x:,} {'(ATM)' if x == atm_strike else ''}"
        )
    with mo_c2:
        option_type = st.radio("Type", ["CE", "PE"], horizontal=True, key="mo_type")
    with mo_c3:
        lots = st.number_input("Lots", min_value=1, max_value=8, value=1, key="mo_lots")
    with mo_c4:
        side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="mo_side")

    qty = int(lots) * 65
    symbol = f"NIFTY{_expiry_str}{selected_strike}{option_type}"

    st.markdown(
        f'<div style="background:var(--card);border:1px solid var(--border);border-radius:6px;'
        f'padding:8px 14px;margin:6px 0;font-size:0.85rem;">'
        f'<span style="color:var(--muted);">Symbol:</span> '
        f'<span class="mono" style="color:var(--text);font-weight:700;">{symbol}</span>'
        f' &bull; <span style="color:var(--muted);">Qty:</span> '
        f'<span class="mono" style="color:var(--text);">{qty}</span>'
        f' &bull; <span style="color:var(--muted);">Expiry:</span> '
        f'<span class="mono" style="color:var(--text);">{_expiry.strftime("%d %b %Y")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2 = st.columns([3, 1])

    with btn_c1:
        place_disabled = not engine_running
        if st.button(
            f"{'BUY' if side == 'BUY' else 'SELL'} {symbol} x{qty}",
            type="primary", use_container_width=True,
            disabled=place_disabled, key="mo_place",
        ):
            from dashboard.data_bridge import MANUAL_ORDER_FILE
            # Check if a previous order is still pending
            existing = DashboardStateReader.read_manual_order()
            if existing.get("status") in ("PENDING", "EXECUTING"):
                st.warning("Previous order still processing...")
            else:
                request = {
                    "id": str(uuid.uuid4()),
                    "status": "PENDING",
                    "requested_at": datetime.now().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "lots": int(lots),
                    "product": "MIS",
                    "order_type": "MARKET",
                    "strike": selected_strike,
                    "option_type": option_type,
                    "result": None,
                    "executed_at": None,
                    "order_id": None,
                    "fill_price": None,
                    "error": None,
                }
                tmp = MANUAL_ORDER_FILE.with_suffix(".tmp")
                with open(tmp, "w") as _mf:
                    json.dump(request, _mf, default=str)
                tmp.replace(MANUAL_ORDER_FILE)
                st.toast(f"Order submitted: {side} {symbol} x{qty}")
                time.sleep(0.5)
                st.rerun()

    with btn_c2:
        if st.button("Clear", use_container_width=True, key="mo_clear"):
            from dashboard.data_bridge import MANUAL_ORDER_FILE
            MANUAL_ORDER_FILE.unlink(missing_ok=True)
            st.rerun()

    # Show order status
    mo_status = DashboardStateReader.read_manual_order()
    if mo_status:
        mo_st = mo_status.get("status", "")
        mo_sym = mo_status.get("symbol", "")
        mo_oid = mo_status.get("order_id", "")
        mo_err = mo_status.get("error", "")
        mo_at = mo_status.get("executed_at", "")

        if mo_st == "PENDING":
            st.markdown(
                f'<div style="padding:8px 14px;background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);'
                f'border-radius:6px;font-size:0.82rem;color:#f59e0b;">PENDING — waiting for engine to execute {mo_sym}...</div>',
                unsafe_allow_html=True,
            )
        elif mo_st == "EXECUTING":
            st.markdown(
                f'<div style="padding:8px 14px;background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.3);'
                f'border-radius:6px;font-size:0.82rem;color:#06b6d4;">EXECUTING — placing order for {mo_sym}...</div>',
                unsafe_allow_html=True,
            )
        elif mo_st == "PLACED":
            st.markdown(
                f'<div style="padding:8px 14px;background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);'
                f'border-radius:6px;font-size:0.82rem;color:#10b981;">'
                f'ORDER PLACED — {mo_sym} | Order ID: {mo_oid} | {mo_at[:19] if mo_at else ""}</div>',
                unsafe_allow_html=True,
            )
        elif mo_st == "FAILED":
            st.markdown(
                f'<div style="padding:8px 14px;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);'
                f'border-radius:6px;font-size:0.82rem;color:#ef4444;">'
                f'FAILED — {mo_sym} | {mo_err}</div>',
                unsafe_allow_html=True,
            )

    if place_disabled:
        st.caption("Start the engine to place orders")


# ══════════════════════════════════════════════════════════════════════════
# RISK & PERFORMANCE DASHBOARD (full width, expandable)
# ══════════════════════════════════════════════════════════════════════════

with st.expander("Risk & Performance", expanded=False):

    rk1, rk2, rk3, rk4, rk5, rk6, rk7, rk8 = st.columns(8)
    with rk1: st.metric("Daily Loss", f"{daily_loss_pct:.2%}")
    with rk2: st.metric("Kill Limit", f"{kill_pct:.0%}")
    with rk3:
        kill_display = "TRIGGERED" if kill_triggered else "SAFE"
        st.metric("Kill Switch", kill_display)
    with rk4: st.metric("Positions", f"{open_count}/{max_pos}")
    with rk5: st.metric("Trades", f"{trades_today}/{max_trades}")
    with rk6: st.metric("Winners", f"{winners}")
    with rk7: st.metric("Losers", f"{losers}")
    with rk8: st.metric("Win Rate", f"{wr}%")

    # Drawdown calculation from P&L curve
    if curve_data and len(curve_data) > 2:
        pnl_series = pd.Series([d["pnl"] for d in curve_data])
        peak = pnl_series.cummax()
        drawdown = pnl_series - peak
        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1]

        dd1, dd2, dd3, dd4 = st.columns(4)
        with dd1: st.metric("Max Drawdown", f"{fmt_inr(max_dd)}")
        with dd2: st.metric("Current DD", f"{fmt_inr(current_dd)}")
        with dd3:
            profit_factor = "N/A"
            if closed_pos:
                gross_profit = sum(p["pnl"] for p in closed_pos if p.get("pnl", 0) > 0)
                gross_loss = abs(sum(p["pnl"] for p in closed_pos if p.get("pnl", 0) < 0))
                profit_factor = f"{gross_profit / gross_loss:.2f}" if gross_loss > 0 else "Inf"
            st.metric("Profit Factor", profit_factor)
        with dd4:
            avg_trade = sum(p.get("pnl", 0) for p in closed_pos) / len(closed_pos) if closed_pos else 0
            st.metric("Avg Trade", f"{fmt_inr(avg_trade)}")


# ══════════════════════════════════════════════════════════════════════════
# ENGINE LOGS
# ══════════════════════════════════════════════════════════════════════════

with st.expander("Engine Logs"):
    logs = state.get("logs", [])
    if logs:
        st.code("\n".join(logs[-40:]), language="log")
    else:
        file_logs = DashboardStateReader.get_recent_logs(40)
        if file_logs:
            st.code("\n".join(file_logs), language="log")
        else:
            st.caption("No logs available")


# ══════════════════════════════════════════════════════════════════════════
# DAILY REPORTS
# ══════════════════════════════════════════════════════════════════════════

with st.expander("Daily Reports"):
    reports = DashboardStateReader.get_daily_reports()
    if reports:
        for rpt in reports[:7]:
            rpt_date = rpt.get("date", "?")
            rpt_pnl = rpt.get("total_pnl", 0)
            rpt_trades = rpt.get("total_trades", 0)
            rpt_wr = rpt.get("win_rate", 0)
            st.markdown(f"""
            <div class="pos-card" style="border-left:3px solid {pnl_color(rpt_pnl)};">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:var(--text);font-weight:700;">{rpt_date}</span>
                    <span class="mono" style="color:{pnl_color(rpt_pnl)};font-weight:800;">
                        {fmt_inr_html(rpt_pnl)}
                    </span>
                </div>
                <div style="color:var(--muted);font-size:0.72rem;margin-top:4px;">
                    {rpt_trades} trades &bull; {rpt_wr:.0f}% win rate
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No daily reports yet")


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="text-align:center;color:var(--muted);font-size:0.62rem;padding:16px 0 8px;
            border-top:1px solid var(--border);margin-top:16px;">
    Aligner V14 Production &bull; Updated: {state.get('last_updated','N/A')[:19]}
    &bull; Refresh: {refresh_speed} &bull; Age: {staleness:.0f}s
    &bull; {len(open_pos)} open &bull; {len(closed_pos)} closed &bull; {len(orders)} orders
</div>""", unsafe_allow_html=True)

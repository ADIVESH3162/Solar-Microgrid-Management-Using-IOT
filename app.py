"""
=====================================================================================
AI-Based Solar Microgrid Management System Using IoT & Machine Learning
Full-Stack Real-Time Dashboard (Streamlit & Plotly)
Supports:
1. Live Hardware Mode: Genuine ESP32 sensor telemetry exclusively logged to solar_dataset_live.csv
2. Simulation Demo Mode: Continuously changing real-time data simulator logged to solar_dataset_demo.csv
=====================================================================================
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alerts import evaluate_telemetry_alerts
from src.config import (
    CSV_COLUMNS,
    DATA_SOURCE_DEMO,
    DATA_SOURCE_HARDWARE,
    DATA_SOURCE_SIMULATION_DEMO,
    DEFAULT_LOAD_POWER_W,
    DEFAULT_MODEL_PATH,
    DEMO_CSV_COLUMNS,
    DEMO_CSV_PATH,
    HISTORICAL_CSV_PATH,
    LIVE_CSV_COLUMNS,
    LIVE_CSV_PATH,
    METRICS_JSON_PATH,
    SAMPLE_CSV_PATH,
)
from src.csv_logger import CSVLogger
from src.data_processor import (
    clean_and_prepare_timeseries,
    compute_cumulative_energy_wh,
    compute_daily_summary,
    get_system_kpis,
)
from src.demo_data_generator import DemoDataGenerator
from src.energy_management import EnergyManager, MicrogridStatus
from src.predict_power import SolarPowerPredictor
from src.sensor_serial import (
    SerialSensorReader,
    detect_esp32_port,
    list_available_ports,
)

# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(
    page_title="AI Solar Microgrid Management System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- CUSTOM CSS STYLING --------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .badge-live {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-sim {
        background-color: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-offline {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-status {
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    .status-sufficient {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .status-insufficient {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .status-unknown {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .demo-banner {
        background-color: #fef3c7;
        color: #92400e;
        border: 2px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 18px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 15px;
    }
    .disconnected-banner {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 12px 18px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SESSION STATE MANAGEMENT --------------
if "logger" not in st.session_state:
    st.session_state.logger = CSVLogger(target_filepath=LIVE_CSV_PATH, columns=LIVE_CSV_COLUMNS)
if "demo_logger" not in st.session_state:
    st.session_state.demo_logger = CSVLogger(target_filepath=DEMO_CSV_PATH, columns=DEMO_CSV_COLUMNS)
if "serial_reader" not in st.session_state:
    st.session_state.serial_reader = SerialSensorReader()
if "demo_generator" not in st.session_state:
    st.session_state.demo_generator = DemoDataGenerator(interval_sec=2.0)
    st.session_state.demo_generator.is_running = True
if "predictor" not in st.session_state:
    st.session_state.predictor = SolarPowerPredictor()
if "has_charge_controller" not in st.session_state:
    st.session_state.has_charge_controller = False
if "sim_is_running" not in st.session_state:
    st.session_state.sim_is_running = True
if "sim_interval" not in st.session_state:
    st.session_state.sim_interval = 2
if "sim_time_window" not in st.session_state:
    st.session_state.sim_time_window = 50

# ----------------- SIDEBAR CONTROLS ----------------------
st.sidebar.image("https://img.icons8.com/color/96/solar-panel.png", width=64)
st.sidebar.title("Microgrid Control")

# 1. Operating Mode Selection
st.sidebar.subheader("System Operating Mode")
operating_mode = st.sidebar.radio(
    "Select Mode:",
    ["🧪 Simulation Demo Mode", "🟢 Live Hardware Mode"],
    index=0,  # Default: Simulation Demo Mode (continuously changes values every 2s)
    help="Simulation Demo Mode continuously generates realistic changing values for demonstrations. Live Hardware Mode reads genuine ESP32 sensors.",
)
is_live_mode = operating_mode == "🟢 Live Hardware Mode"
is_demo_mode = not is_live_mode

# Navigation menu
PAGE_OPTIONS = [
    "🏠 1. Home Dashboard",
    "⚡ 2. Live Sensor Monitoring",
    "📈 3. Solar Generation Analysis",
    "🗃️ 4. CSV Dataset Explorer",
    "📊 5. Historical Data Analysis",
    "🤖 6. Machine Learning Forecast",
    "⚖️ 7. Energy Management & Load",
    "🩺 8. Sensor & System Health",
    "📘 9. Project Documentation",
]
selected_page = st.sidebar.radio("Navigation", PAGE_OPTIONS)

st.sidebar.divider()

# Mode-Specific Sidebar Controls
if is_live_mode:
    st.sidebar.subheader("ESP32 Hardware Connection")
    available_ports = list_available_ports()
    port_options = [p["device"] for p in available_ports]
    detected_esp = detect_esp32_port()

    if detected_esp and detected_esp not in port_options:
        port_options.append(detected_esp)

    selected_port = st.sidebar.selectbox(
        "ESP32 Serial Port (115200 baud)",
        options=["AUTO"] + port_options if port_options else ["None detected"],
        index=0,
        help="Select USB COM port for ESP32 Dev Board. 'AUTO' searches for CP210x/CH340 chips.",
    )

    col_conn1, col_conn2 = st.sidebar.columns(2)
    with col_conn1:
        if st.button("🔌 Connect", use_container_width=True):
            target_p = None if selected_port == "AUTO" else selected_port
            st.session_state.serial_reader.port = target_p
            if st.session_state.serial_reader.connect(port=target_p, csv_logger=st.session_state.logger):
                st.sidebar.success(f"Connected to {st.session_state.serial_reader.port}")
            else:
                st.sidebar.error(st.session_state.serial_reader.last_error or "Connection failed")

    with col_conn2:
        if st.button("❌ Disconnect", use_container_width=True):
            st.session_state.serial_reader.disconnect()
            st.sidebar.info("Disconnected")

    if st.session_state.serial_reader.is_connected:
        st.sidebar.markdown(f"**Status:** 🟢 Connected (`{st.session_state.serial_reader.port}`)")
        st.sidebar.caption(f"Valid Readings: **{st.session_state.serial_reader.valid_reading_count:,}**")
    else:
        st.sidebar.markdown("**Status:** 🔴 Disconnected")

else:
    # ----------------- SIMULATION DEMO CONTROLS -----------------
    st.sidebar.subheader("Simulation Demo Controls")

    # Update interval selector: 1, 2, 5, 10 seconds
    interval_options = [1, 2, 5, 10]
    curr_idx = interval_options.index(st.session_state.sim_interval) if st.session_state.sim_interval in interval_options else 1
    selected_interval = st.sidebar.selectbox(
        "Update Interval:",
        interval_options,
        index=curr_idx,
        format_func=lambda x: f"{x} second{'s' if x > 1 else ''}",
        help="Configurable interval at which new simulated readings are generated.",
    )
    if selected_interval != st.session_state.sim_interval:
        st.session_state.sim_interval = selected_interval
        st.session_state.demo_generator.set_interval(selected_interval)

    # Time window selector for graphs
    st.session_state.sim_time_window = st.sidebar.selectbox(
        "Graph Time Window:",
        [20, 50, 100, 200, 500],
        index=[20, 50, 100, 200, 500].index(st.session_state.sim_time_window) if st.session_state.sim_time_window in [20, 50, 100, 200, 500] else 1,
        format_func=lambda x: f"Last {x} readings",
        help="Number of recent readings displayed across time series graphs.",
    )

    # Start / Stop simulation controls
    col_sim_btn1, col_sim_btn2 = st.sidebar.columns(2)
    with col_sim_btn1:
        if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.sim_is_running):
            st.session_state.sim_is_running = True
            st.session_state.demo_generator.is_running = True
            st.rerun()

    with col_sim_btn2:
        if st.button("⏸️ Stop", use_container_width=True, disabled=not st.session_state.sim_is_running):
            st.session_state.sim_is_running = False
            st.session_state.demo_generator.is_running = False
            st.rerun()

    if st.session_state.sim_is_running:
        st.sidebar.markdown(f"**Simulator:** 🟢 RUNNING (Every {st.session_state.sim_interval}s)")
    else:
        st.sidebar.markdown("**Simulator:** ⏸️ STOPPED")

    # Clear demo data with confirmation
    with st.sidebar.expander("🗑️ Clear Demo Data"):
        confirm_clear = st.checkbox("Confirm clearing demo session data and CSV log", key="confirm_clear_demo")
        if st.button("Clear Demo Data Now", disabled=not confirm_clear, use_container_width=True):
            st.session_state.demo_generator.clear_history(DEMO_CSV_PATH)
            st.session_state.demo_logger.clear_file()
            st.sidebar.success("Demo dataset cleared.")
            st.rerun()

# DC Load configuration
st.sidebar.divider()
st.sidebar.subheader("Microgrid Load Settings")
configured_load_w = st.sidebar.slider(
    "Target DC Load Demand (Watts)",
    min_value=0.5,
    max_value=20.0,
    value=float(DEFAULT_LOAD_POWER_W),
    step=0.5,
    help="Target load power (e.g. 5W DC LED light, micro-inverter, or test resistor load).",
)

st.session_state.has_charge_controller = st.sidebar.checkbox(
    "Solar Charge Controller Verified",
    value=st.session_state.has_charge_controller,
    help="Check ONLY if a dedicated solar charge controller (e.g. TP4056 with CC/CV or MPPT) is wired between solar and 18650 cell.",
)

# ----------------- DATASET & TELEMETRY RESOLUTION --------
if is_live_mode:
    # LIVE HARDWARE MODE: Exclusively query solar_dataset_live.csv and serial reader
    raw_live_df = st.session_state.logger.load_dataframe()
    if not raw_live_df.empty:
        active_df = clean_and_prepare_timeseries(raw_live_df)
        active_source_label = f"Live Hardware Log ({len(active_df):,} genuine records)"
    else:
        active_df = pd.DataFrame(columns=LIVE_CSV_COLUMNS)
        active_source_label = "Live Hardware Log (0 records logged yet)"

    is_connected = st.session_state.serial_reader.is_connected
    if is_connected:
        lr = st.session_state.serial_reader.get_latest_reading()
        conn_badge = f'<span class="badge-live">● ESP32 CONNECTED ({st.session_state.serial_reader.port})</span>'
    else:
        lr = None
        conn_badge = '<span class="badge-offline">● DISCONNECTED — WAITING FOR REAL SENSOR DATA</span>'

    kpis = get_system_kpis(active_df, default_load_w=configured_load_w)

    if lr:
        curr_p = lr.get("solar_power_w")
        curr_v = lr.get("solar_voltage_v")
        curr_i = lr.get("solar_current_ma")
        curr_lux = lr.get("light_lux")
        curr_t = lr.get("temperature_c")
        curr_h = lr.get("humidity_pct")
        last_ts = lr.get("timestamp")
        sensor_st = lr.get("sensor_status", "OK")

        kpis["current_power_w"] = curr_p
        kpis["current_voltage_v"] = curr_v
        kpis["current_current_ma"] = curr_i
        kpis["current_lux"] = curr_lux
        kpis["current_temp_c"] = curr_t
        kpis["current_humidity_pct"] = curr_h
        kpis["last_timestamp"] = str(last_ts)
        kpis["sensor_status"] = sensor_st

        if curr_p is not None:
            kpis["surplus_deficit_w"] = round(curr_p - configured_load_w, 3)
            kpis["solar_sufficiency"] = (
                "SOLAR_SUFFICIENT" if curr_p >= configured_load_w else "SOLAR_INSUFFICIENT"
            )
        else:
            kpis["surplus_deficit_w"] = None
            kpis["solar_sufficiency"] = "AWAITING_DATA"
    elif not is_connected:
        kpis["current_power_w"] = None
        kpis["current_voltage_v"] = None
        kpis["current_current_ma"] = None
        kpis["current_lux"] = None
        kpis["current_temp_c"] = None
        kpis["current_humidity_pct"] = None
        kpis["solar_sufficiency"] = "DISCONNECTED"
        kpis["surplus_deficit_w"] = None
        kpis["sensor_status"] = "DISCONNECTED"
        if active_df.empty:
            kpis["last_timestamp"] = "No Real Data Yet"

else:
    # SIMULATION DEMO MODE: Continuously generate changing values
    conn_badge = (
        f'<span class="badge-sim">● SIMULATION DEMO {"(RUNNING)" if st.session_state.sim_is_running else "(STOPPED)"}</span>'
    )

    # Step simulator if running
    if st.session_state.sim_is_running:
        st.session_state.demo_generator.is_running = True
        st.session_state.demo_generator.set_interval(st.session_state.sim_interval)
        lr = st.session_state.demo_generator.step(csv_logger=st.session_state.demo_logger, force=True)
    else:
        st.session_state.demo_generator.is_running = False
        lr = st.session_state.demo_generator.get_latest_reading()

    # Load recent history from demo CSV for continuous timeline
    raw_demo_df = st.session_state.demo_logger.load_dataframe()
    if not raw_demo_df.empty:
        active_df = clean_and_prepare_timeseries(raw_demo_df)
        active_source_label = f"Simulation Demo Log ({len(active_df):,} records)"
    else:
        active_df = st.session_state.demo_generator.get_history_df()
        active_source_label = f"Simulation Demo In-Memory ({len(active_df):,} records)"

    kpis = get_system_kpis(active_df, default_load_w=configured_load_w)

    if lr:
        curr_p = lr.get("solar_power_w")
        curr_v = lr.get("solar_voltage_v")
        curr_i = lr.get("solar_current_ma")
        curr_lux = lr.get("light_lux")
        curr_t = lr.get("temperature_c")
        curr_h = lr.get("humidity_pct")
        last_ts = lr.get("timestamp")

        kpis["current_power_w"] = curr_p
        kpis["current_voltage_v"] = curr_v
        kpis["current_current_ma"] = curr_i
        kpis["current_lux"] = curr_lux
        kpis["current_temp_c"] = curr_t
        kpis["current_humidity_pct"] = curr_h
        kpis["last_timestamp"] = str(last_ts)
        kpis["sensor_status"] = "SIMULATED_DEMO"

        if curr_p is not None:
            kpis["surplus_deficit_w"] = round(curr_p - configured_load_w, 3)
            kpis["solar_sufficiency"] = (
                "SOLAR_SUFFICIENT" if curr_p >= configured_load_w else "SOLAR_INSUFFICIENT"
            )

# Energy management decision
energy_mgr = EnergyManager(load_power_w=configured_load_w)
energy_decision = energy_mgr.evaluate_energy_balance(
    solar_power_w=kpis["current_power_w"],
    load_power_w=configured_load_w,
    battery_voltage_v=None,
    has_charge_controller=st.session_state.has_charge_controller,
)

# Helper to format measurements
def fmt_val(val, unit="", decimals=2, missing_label="N/A"):
    if val is None or pd.isna(val) or val == "":
        return missing_label
    try:
        f = float(val)
        if decimals == 0:
            return f"{f:,.0f} {unit}".strip()
        return f"{f:.{decimals}f} {unit}".strip()
    except (ValueError, TypeError):
        return missing_label


# =========================================================
# PAGE 1: HOME DASHBOARD
# =========================================================
if "1. Home Dashboard" in selected_page:
    st.markdown('<div class="main-header">☀️ Solar Microgrid Management System</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">AI-Powered Photovoltaic Telemetry & Load Balancing &nbsp;|&nbsp; {conn_badge}</div>',
        unsafe_allow_html=True,
    )

    # Prominent Status Banners
    if is_live_mode:
        if not st.session_state.serial_reader.is_connected:
            st.markdown(
                """
                <div class="disconnected-banner">
                    ⚠️ DISCONNECTED — WAITING FOR REAL SENSOR DATA<br>
                    <span style="font-size:0.92rem; font-weight:normal;">
                    ESP32 hardware is not connected. Connect your ESP32 board over USB, select COM port in the sidebar, and click <b>Connect</b>.
                    Or switch to <b>Simulation Demo Mode</b> in the sidebar for automated demonstration.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="demo-banner">
                ⚠️ SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS<br>
                <span style="font-size:0.92rem; font-weight:normal;">
                Continuously changing simulated values are being generated. All records are stored separately in <b>data/solar_dataset_demo.csv</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top KPI row: Power, Voltage, Current, Cumulative Energy
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        p_str = fmt_val(kpis["current_power_w"], unit="W", decimals=3)
        delta_str = f"{kpis['surplus_deficit_w']:+.3f} W vs Load" if kpis["surplus_deficit_w"] is not None else None
        st.metric("Solar Power", p_str, delta=delta_str, help="Power (W) = Voltage (V) × Current (A)")
    with col2:
        v_str = fmt_val(kpis["current_voltage_v"], unit="V", decimals=2)
        st.metric("Solar Voltage", v_str, help="Measured / Simulated DC bus voltage")
    with col3:
        i_str = fmt_val(kpis["current_current_ma"], unit="mA", decimals=1)
        st.metric("Solar Current", i_str, help="Measured / Simulated DC current in mA")
    with col4:
        st.metric("Cumulative Energy", f"{kpis['total_energy_wh']:.3f} Wh", help="Numerical trapezoidal integration from logged records")

    # Second KPI row: Lux, Temp, Humidity, Valid Records
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        lux_str = fmt_val(kpis["current_lux"], unit="lx", decimals=0)
        st.metric("Light Intensity", lux_str, help="Sunlight Lux")
    with col6:
        t_str = fmt_val(kpis["current_temp_c"], unit="°C", decimals=1)
        st.metric("Ambient Temperature", t_str, help="Ambient Temperature")
    with col7:
        st.metric("Target Load Demand", f"{configured_load_w:.1f} W")
    with col8:
        if is_live_mode:
            rec_count = st.session_state.logger.get_record_count()
            st.metric("Genuine Records Logged", f"{rec_count:,}")
        else:
            rec_count = st.session_state.demo_logger.get_record_count()
            st.metric("Demo Records Generated", f"{rec_count:,}")

    st.divider()

    # Sufficiency & Recommendation Banner
    col_status, col_advice = st.columns([1, 2])
    with col_status:
        st.subheader("Microgrid Power Status")
        if energy_decision["status"] == MicrogridStatus.SOLAR_SUFFICIENT.value:
            st.markdown('<div class="badge-status status-sufficient">✅ SOLAR SUFFICIENT</div>', unsafe_allow_html=True)
            st.write(f"**Surplus Generation:** +{energy_decision['surplus_w']:.2f} W")
        elif energy_decision["status"] == MicrogridStatus.SOLAR_INSUFFICIENT.value:
            st.markdown('<div class="badge-status status-insufficient">⚠️ SOLAR INSUFFICIENT</div>', unsafe_allow_html=True)
            st.write(f"**Deficit Shortfall:** -{energy_decision['deficit_w']:.2f} W")
        else:
            st.markdown('<div class="badge-status status-unknown">⏳ WAITING FOR REAL SENSOR DATA</div>', unsafe_allow_html=True)
            st.write("Awaiting live measurements from ESP32...")

    with col_advice:
        st.subheader("Energy Dispatch Recommendation")
        st.info(f"**Load Action:** {energy_decision['action_recommendation']}")
        st.warning(f"**Storage Advisory:** {energy_decision['battery_advisory']}")

    # Real-Time Visual Charts
    st.subheader(f"Recent Generation Trend ({active_source_label})")
    if not active_df.empty and "solar_power_w" in active_df.columns:
        recent_df = active_df.tail(st.session_state.sim_time_window if is_demo_mode else 60)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=recent_df["timestamp"], y=recent_df["solar_power_w"],
            mode="lines+markers", name="Solar Power (W)", line=dict(color="#f59e0b", width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=recent_df["timestamp"], y=[configured_load_w]*len(recent_df),
            mode="lines", name="Configured Load Demand (W)", line=dict(color="#ef4444", dash="dash")
        ))
        fig_trend.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            xaxis_title="Timestamp",
            yaxis_title="Power (Watts)",
            template="plotly_white",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        if is_live_mode:
            st.info("ℹ️ No genuine sensor measurements logged yet. Connect ESP32 to start streaming real data.")
        else:
            st.info("ℹ️ Generating first simulation readings...")


# =========================================================
# PAGE 2: LIVE SENSOR MONITORING & CONTINUOUS GRAPHS
# =========================================================
elif "2. Live Sensor Monitoring" in selected_page:
    st.markdown('<div class="main-header">⚡ Live Sensor Monitoring</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Direct Real-Time Telemetry &nbsp;|&nbsp; {conn_badge}</div>', unsafe_allow_html=True)

    if is_live_mode:
        if not st.session_state.serial_reader.is_connected:
            st.markdown(
                """
                <div class="disconnected-banner">
                    ⚠️ DISCONNECTED — WAITING FOR REAL SENSOR DATA<br>
                    <span style="font-size:0.92rem; font-weight:normal;">
                    Hardware is currently offline. Connect your ESP32 dev board via USB to display genuine sensor readings.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success(
                f"✅ **ESP32 Live Stream Active** | Port: `{st.session_state.serial_reader.port}` | "
                f"Last Received: `{kpis.get('last_timestamp', 'Just now')}` | "
                f"Valid Readings: `{st.session_state.serial_reader.valid_reading_count:,}`"
            )
    else:
        st.markdown(
            """
            <div class="demo-banner">
                ⚠️ SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS<br>
                <span style="font-size:0.92rem; font-weight:normal;">
                Continuously updating simulated sensor readings at <b>""" + str(st.session_state.sim_interval) + """s</b> intervals.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sensor Gauges Row
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        val_p = kpis["current_power_w"] if kpis["current_power_w"] is not None else 0.0
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_p,
            title={"text": "Solar Power (W)" if kpis["current_power_w"] is not None else "Solar Power (N/A)"},
            gauge={"axis": {"range": [0, 15]}, "bar": {"color": "#f59e0b" if kpis["current_power_w"] is not None else "#cbd5e1"}},
        ))
        fig_g1.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_g1, use_container_width=True)

    with col_g2:
        val_v = kpis["current_voltage_v"] if kpis["current_voltage_v"] is not None else 0.0
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_v,
            title={"text": "Solar Voltage (V)" if kpis["current_voltage_v"] is not None else "Solar Voltage (N/A)"},
            gauge={"axis": {"range": [0, 25]}, "bar": {"color": "#3b82f6" if kpis["current_voltage_v"] is not None else "#cbd5e1"}},
        ))
        fig_g2.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_g2, use_container_width=True)

    with col_g3:
        val_lux = kpis["current_lux"] if kpis["current_lux"] is not None else 0.0
        fig_g3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_lux,
            title={"text": "Sunlight (Lux)" if kpis["current_lux"] is not None else "Sunlight (N/A)"},
            gauge={"axis": {"range": [0, 100000]}, "bar": {"color": "#10b981" if kpis["current_lux"] is not None else "#cbd5e1"}},
        ))
        fig_g3.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_g3, use_container_width=True)

    # ----------------- CONTINUOUSLY UPDATING PLOTLY GRAPHS -----------------
    st.subheader(f"Multi-Parameter Time Series ({'Simulation Window: ' + str(st.session_state.sim_time_window) + ' pts' if is_demo_mode else active_source_label})")

    if not active_df.empty:
        window_size = st.session_state.sim_time_window if is_demo_mode else 100
        df_plot = active_df.tail(window_size)

        # 6 Parameter Graphs Display
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            # Graph 1: Solar Power (W)
            if "solar_power_w" in df_plot.columns:
                fig_p = px.line(df_plot, x="timestamp", y="solar_power_w", title="Solar Power (Watts)", markers=True, color_discrete_sequence=["#f59e0b"])
                fig_p.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_white")
                st.plotly_chart(fig_p, use_container_width=True)

            # Graph 2: Solar Voltage (V)
            if "solar_voltage_v" in df_plot.columns:
                fig_v = px.line(df_plot, x="timestamp", y="solar_voltage_v", title="Solar Voltage (Volts)", markers=True, color_discrete_sequence=["#3b82f6"])
                fig_v.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_white")
                st.plotly_chart(fig_v, use_container_width=True)

            # Graph 3: Ambient Temperature (°C)
            if "temperature_c" in df_plot.columns:
                fig_t = px.line(df_plot, x="timestamp", y="temperature_c", title="Ambient Temperature (°C)", markers=True, color_discrete_sequence=["#ef4444"])
                fig_t.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_white")
                st.plotly_chart(fig_t, use_container_width=True)

        with col_p2:
            # Graph 4: Solar Current (mA)
            if "solar_current_ma" in df_plot.columns:
                fig_i = px.line(df_plot, x="timestamp", y="solar_current_ma", title="Solar Current (mA)", markers=True, color_discrete_sequence=["#06b6d4"])
                fig_i.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_white")
                st.plotly_chart(fig_i, use_container_width=True)

            # Graph 5: Sunlight Intensity (Lux)
            if "light_lux" in df_plot.columns:
                fig_l = px.area(df_plot, x="timestamp", y="light_lux", title="Sunlight Intensity (Lux)", color_discrete_sequence=["#10b981"])
                fig_l.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_white")
                st.plotly_chart(fig_l, use_container_width=True)

            # Graph 6: Relative Humidity (%)
            if "humidity_pct" in df_plot.columns:
                fig_h = px.line(df_plot, x="timestamp", y="humidity_pct", title="Relative Humidity (%)", markers=True, color_discrete_sequence=["#8b5cf6"])
                fig_h.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), template="plotly_white")
                st.plotly_chart(fig_h, use_container_width=True)

    else:
        st.info("ℹ️ No readings recorded yet.")

    # ----------------- CSV DATASET VIEWER & DOWNLOAD -----------------
    st.divider()
    if is_live_mode:
        st.subheader("Live Dataset Stream (`solar_dataset_live.csv`)")
        col_ld1, col_ld2 = st.columns([3, 1])
        with col_ld1:
            st.write("Recent rows recorded directly from ESP32 hardware:")
        with col_ld2:
            if LIVE_CSV_PATH.exists():
                st.download_button(
                    label="📥 Download Live CSV",
                    data=LIVE_CSV_PATH.read_bytes(),
                    file_name="solar_dataset_live.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        df_live_view = st.session_state.logger.load_dataframe()
        if not df_live_view.empty:
            st.dataframe(df_live_view.tail(20), use_container_width=True)
        else:
            st.info("ℹ️ Live dataset is currently empty (0 records). Connect ESP32 to record genuine measurements.")

    else:
        st.subheader("Simulation Demo Dataset Stream (`solar_dataset_demo.csv`)")
        col_dd1, col_dd2 = st.columns([3, 1])
        with col_dd1:
            st.write(f"Recent records saved separately to `{DEMO_CSV_PATH.name}` (data_source=simulation_demo):")
        with col_dd2:
            if DEMO_CSV_PATH.exists():
                st.download_button(
                    label="📥 Download Demo CSV",
                    data=DEMO_CSV_PATH.read_bytes(),
                    file_name="solar_dataset_demo.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        df_demo_view = st.session_state.demo_logger.load_dataframe()
        if not df_demo_view.empty:
            st.dataframe(df_demo_view.tail(20), use_container_width=True)
        else:
            st.info("ℹ️ Demo dataset is currently empty. Start simulation to generate readings.")


# =========================================================
# PAGE 3: SOLAR ENERGY GENERATION
# =========================================================
elif "3. Solar Generation Analysis" in selected_page:
    st.markdown('<div class="main-header">📈 Solar Energy Generation Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Cumulative Watt-Hours (Wh) & Diurnal PV Yield Profiles &nbsp;|&nbsp; {conn_badge}</div>', unsafe_allow_html=True)

    if not is_live_mode:
        st.markdown('<div class="demo-banner">⚠️ SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS</div>', unsafe_allow_html=True)

    if not active_df.empty and "solar_power_w" in active_df.columns:
        df_energy = compute_cumulative_energy_wh(active_df)

        col_e1, col_e2, col_e3 = st.columns(3)
        total_wh = df_energy["energy_wh"].iloc[-1] if "energy_wh" in df_energy and not df_energy.empty else 0.0
        peak_w = df_energy["solar_power_w"].dropna().max() if "solar_power_w" in df_energy and not df_energy["solar_power_w"].dropna().empty else 0.0
        avg_w = df_energy["solar_power_w"].dropna().mean() if "solar_power_w" in df_energy and not df_energy["solar_power_w"].dropna().empty else 0.0

        with col_e1:
            st.metric("Total Cumulative Energy", f"{total_wh:.3f} Wh", f"{(total_wh/1000.0):.5f} kWh")
        with col_e2:
            st.metric("Peak Generation Power", f"{peak_w:.3f} W")
        with col_e3:
            st.metric("Average Generation Power", f"{avg_w:.3f} W")

        st.subheader("Cumulative Energy Yield Curve (Watt-Hours)")
        fig_yield = px.area(df_energy, x="timestamp", y="energy_wh", color_discrete_sequence=["#10b981"])
        fig_yield.update_layout(height=350, template="plotly_white", yaxis_title="Energy (Wh)")
        st.plotly_chart(fig_yield, use_container_width=True)

        st.subheader("Daily Generation Summary")
        df_daily = compute_daily_summary(active_df)
        if not df_daily.empty:
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                fig_daily_bar = px.bar(
                    df_daily, x="date", y="total_energy_wh",
                    title="Daily Energy Generation (Wh)",
                    color="total_energy_wh", color_continuous_scale="Greens"
                )
                fig_daily_bar.update_layout(template="plotly_white", height=300)
                st.plotly_chart(fig_daily_bar, use_container_width=True)
            with col_d2:
                st.dataframe(df_daily, use_container_width=True)
    else:
        st.info("ℹ️ Awaiting records to compute cumulative energy.")


# =========================================================
# PAGE 4: CSV DATASET VIEWER & DOWNLOADER
# =========================================================
elif "4. CSV Dataset Explorer" in selected_page:
    st.markdown('<div class="main-header">🗃️ CSV Dataset Explorer & Downloader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect, Validate, and Download Microgrid Datasets</div>', unsafe_allow_html=True)

    dataset_choice = st.radio(
        "Choose Dataset to Inspect:",
        [
            "Live Hardware Telemetry (solar_dataset_live.csv)",
            "Simulation Demo Telemetry (solar_dataset_demo.csv)",
            "Sample Baseline (solar_dataset_sample.csv)",
            "Historical Archive (solar_dataset_historical.csv)",
        ],
        horizontal=True,
    )

    if "Live" in dataset_choice:
        chosen_path = LIVE_CSV_PATH
        desc_label = "Genuine Hardware Dataset (data_source=esp32_hardware)"
        is_demo_file = False
    elif "Simulation" in dataset_choice:
        chosen_path = DEMO_CSV_PATH
        desc_label = "SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS (data_source=simulation_demo)"
        is_demo_file = True
    elif "Sample" in dataset_choice:
        chosen_path = SAMPLE_CSV_PATH
        desc_label = "Synthetic Benchmark Sample"
        is_demo_file = True
    else:
        chosen_path = HISTORICAL_CSV_PATH
        desc_label = "Multi-Day Historical Archive"
        is_demo_file = True

    if is_demo_file:
        st.markdown(f'<div class="demo-banner">ℹ️ Viewing: {desc_label}</div>', unsafe_allow_html=True)

    if chosen_path.exists():
        df_view = pd.read_csv(chosen_path)
        st.write(f"**Loaded File:** `{chosen_path.name}` ({len(df_view):,} rows, {len(df_view.columns)} columns) — *{desc_label}*")

        col_flt1, col_flt2 = st.columns([2, 1])
        with col_flt1:
            records_to_show = st.slider("Display row limit", 10, min(5000, max(20, len(df_view))), 100)
        with col_flt2:
            st.download_button(
                label=f"📥 Download {chosen_path.name}",
                data=chosen_path.read_bytes(),
                file_name=chosen_path.name,
                mime="text/csv",
                use_container_width=True,
            )

        if not df_view.empty:
            st.dataframe(df_view.tail(records_to_show), use_container_width=True)
            with st.expander("📊 Dataset Statistical Summary"):
                st.dataframe(df_view.describe(), use_container_width=True)
        else:
            st.info(f"File `{chosen_path.name}` exists but contains 0 data rows.")
    else:
        st.warning(f"File `{chosen_path.name}` does not exist on disk yet.")


# =========================================================
# PAGE 5: HISTORICAL DATA ANALYSIS
# =========================================================
elif "5. Historical Data Analysis" in selected_page:
    st.markdown('<div class="main-header">📊 Historical Microgrid Telemetry Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cross-Correlation & Weather Impact Analytics</div>', unsafe_allow_html=True)

    if HISTORICAL_CSV_PATH.exists():
        df_hist = pd.read_csv(HISTORICAL_CSV_PATH)
        df_hist = clean_and_prepare_timeseries(df_hist)

        st.write(f"Analyzing `{HISTORICAL_CSV_PATH.name}` ({len(df_hist):,} records across 7 days)")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("Sunlight vs Solar Power Generation")
            fig_scatter1 = px.scatter(
                df_hist, x="light_lux", y="solar_power_w", color="temperature_c",
                labels={"light_lux": "Light (Lux)", "solar_power_w": "Power (W)", "temperature_c": "Temp (°C)"},
                title="Irradiance to Electrical Power Transfer",
                color_continuous_scale="Viridis",
            )
            fig_scatter1.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_scatter1, use_container_width=True)

        with col_h2:
            st.subheader("Thermal Impact on PV Efficiency")
            fig_scatter2 = px.scatter(
                df_hist[df_hist["solar_power_w"] > 0.5],
                x="temperature_c", y="solar_power_w", color="humidity_pct",
                labels={"temperature_c": "Temperature (°C)", "solar_power_w": "Power (W)"},
                title="Temperature Derating Effects",
                color_continuous_scale="Plasma",
            )
            fig_scatter2.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_scatter2, use_container_width=True)

        st.subheader("Multi-Parameter Correlation Heatmap")
        num_cols = ["light_lux", "solar_voltage_v", "solar_current_ma", "solar_power_w", "temperature_c", "humidity_pct"]
        corr_matrix = df_hist[num_cols].corr()
        fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig_heat.update_layout(height=380)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.error(f"Historical dataset `{HISTORICAL_CSV_PATH}` not found.")


# =========================================================
# PAGE 6: MACHINE LEARNING FORECAST
# =========================================================
elif "6. Machine Learning Forecast" in selected_page:
    st.markdown('<div class="main-header">🤖 Machine Learning Solar Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Short-Term PV Power Prediction with Random Forest Regressor</div>', unsafe_allow_html=True)

    predictor = st.session_state.predictor

    if predictor.is_loaded:
        meta = predictor.metadata
        is_syn = meta.get("is_synthetic_demonstration", True)

        if is_syn:
            st.warning(
                "ℹ️ **PROVENANCE NOTICE:** This model was trained on demonstration/synthetic benchmark data. "
                "Reported accuracy metrics reflect proof-of-concept modeling and must be validated against genuine hardware measurements."
            )
        else:
            st.success("✅ **PROVENANCE:** Model trained on genuine physical hardware measurements.")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Model Architecture", meta.get("model_type", "RandomForest"))
        with col_m2:
            st.metric("Mean Absolute Error (MAE)", f"{meta.get('mae_w', 0.0):.4f} W")
        with col_m3:
            st.metric("Root Mean Squared Error (RMSE)", f"{meta.get('rmse_w', 0.0):.4f} W")
        with col_m4:
            st.metric("R² Score", f"{meta.get('r2_score', 0.0):.4f}")

        st.divider()

        st.subheader("Interactive Power Prediction Calculator")
        default_lux = float(kpis["current_lux"]) if kpis["current_lux"] is not None else 45000.0
        default_temp = float(kpis["current_temp_c"]) if kpis["current_temp_c"] is not None else 30.0
        default_hum = float(kpis["current_humidity_pct"]) if kpis["current_humidity_pct"] is not None else 50.0

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            in_lux = st.slider("Input Light Intensity (Lux)", 0.0, 100000.0, default_lux, 500.0)
        with col_in2:
            in_temp = st.slider("Ambient Temperature (°C)", 10.0, 55.0, default_temp, 0.5)
        with col_in3:
            in_hum = st.slider("Relative Humidity (%)", 10.0, 95.0, default_hum, 1.0)

        pred_res = predictor.predict_from_record(light_lux=in_lux, temperature_c=in_temp, humidity_pct=in_hum)
        predicted_w = pred_res["predicted_power_w"]

        st.metric(
            label="Predicted Available Solar Power",
            value=f"{predicted_w:.3f} Watts",
            delta=f"{predicted_w - configured_load_w:+.3f} W vs {configured_load_w:.1f}W Load",
        )
        st.caption(f"Status: {pred_res['warning']}")

        st.divider()
        st.subheader("Prediction vs Actual Ground Truth (Chronological Test Split)")
        if HISTORICAL_CSV_PATH.exists():
            df_test = pd.read_csv(HISTORICAL_CSV_PATH).tail(288)
            df_pred = predictor.predict_dataframe(df_test)

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=df_pred["timestamp"], y=df_pred["solar_power_w"],
                name="Actual Ground Truth (W)", line=dict(color="#10b981", width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=df_pred["timestamp"], y=df_pred["predicted_power_w"],
                name="RF Predicted Power (W)", line=dict(color="#6366f1", dash="dot", width=2)
            ))
            fig_pred.update_layout(
                height=350,
                xaxis_title="Timestamp",
                yaxis_title="Power (Watts)",
                template="plotly_white",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

    else:
        st.error("Model weights not found. Run `python src/train_model.py` or execute `train_model.bat` to train the model.")


# =========================================================
# PAGE 7: ENERGY MANAGEMENT & LOAD RECOMMENDATIONS
# =========================================================
elif "7. Energy Management & Load" in selected_page:
    st.markdown('<div class="main-header">⚖️ Microgrid Energy Management & Dispatch</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Load Matching, Surplus/Deficit Balancing, and Storage Advisory &nbsp;|&nbsp; {conn_badge}</div>', unsafe_allow_html=True)

    col_em1, col_em2 = st.columns([1, 1])

    with col_em1:
        st.subheader("Current Balance Equation")
        p_gen = kpis["current_power_w"]
        p_load = configured_load_w

        if p_gen is not None:
            p_net = p_gen - p_load
            st.write(f"**Generation ($P_{{gen}}$):** {p_gen:.3f} W")
            st.write(f"**Load Demand ($P_{{load}}$):** {p_load:.3f} W")
            st.write(f"**Net Balance ($P_{{net}}$):** {p_net:+.3f} W")

            if p_net >= 0:
                st.success(f"**State:** SOLAR_SUFFICIENT (+{p_net:.2f} W Surplus)")
            else:
                st.error(f"**State:** SOLAR_INSUFFICIENT ({abs(p_net):.2f} W Deficit)")

            fig_bal = go.Figure(go.Bar(
                x=["Solar Generated", "Load Demand"],
                y=[p_gen, p_load],
                marker_color=["#10b981" if p_gen >= p_load else "#f59e0b", "#ef4444"],
                text=[f"{p_gen:.2f} W", f"{p_load:.2f} W"],
                textposition="auto",
            ))
            fig_bal.update_layout(height=280, template="plotly_white", yaxis_title="Power (Watts)")
            st.plotly_chart(fig_bal, use_container_width=True)
        else:
            st.info("ℹ️ Solar generation measurement unavailable. Awaiting readings.")

    with col_em2:
        st.subheader("Energy Storage & Battery Advisory")
        st.info("ℹ️ Dedicated battery terminal sensor is not wired. Battery fields recorded as null per safety standards.")

        st.markdown("#### Hardware Safety Rules:")
        st.markdown(
            """
            1. **Charge Controller Required:** A bare 18650 cell must never be connected to a raw solar panel without CC/CV regulation (e.g. TP4056 with protection or MPPT).
            2. **Advisory Dispatch Mode:** This software calculates recommendations. Physical load shedding requires isolated relays or MOSFET switches.
            3. **Surplus Storage Policy:** Only surplus power exceeding critical load requirements is routed to storage.
            """
        )

        st.warning(f"**System Advisory:** {energy_decision['battery_advisory']}")


# =========================================================
# PAGE 8: SENSOR & SYSTEM HEALTH
# =========================================================
elif "8. Sensor & System Health" in selected_page:
    st.markdown('<div class="main-header">🩺 System & Sensor Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Hardware Link Health, Diagnostics, and Alarms &nbsp;|&nbsp; {conn_badge}</div>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("System Status")
        if is_live_mode:
            is_conn = st.session_state.serial_reader.is_connected
            health_data = [
                {
                    "Component": "ESP32 Controller",
                    "Interface": "USB Serial (115200 baud)",
                    "Status": f"ONLINE ({st.session_state.serial_reader.port})" if is_conn else "DISCONNECTED",
                },
                {
                    "Component": "DHT22 Sensor",
                    "Interface": "GPIO 4 (One-Wire)",
                    "Status": "NOMINAL" if (kpis["current_temp_c"] is not None) else ("AWAITING DATA" if is_conn else "OFFLINE"),
                },
                {
                    "Component": "BH1750 Sensor",
                    "Interface": "I2C (Addr: 0x23, SDA=21, SCL=22)",
                    "Status": "NOMINAL" if (kpis["current_lux"] is not None) else ("AWAITING DATA" if is_conn else "OFFLINE"),
                },
                {
                    "Component": "INA219 Power IC",
                    "Interface": "I2C (Addr: 0x40, SDA=21, SCL=22)",
                    "Status": "NOMINAL" if (kpis["current_voltage_v"] is not None) else ("AWAITING DATA" if is_conn else "OFFLINE"),
                },
            ]
            st.dataframe(pd.DataFrame(health_data), use_container_width=True)
            st.metric("Total Genuine Records Logged", f"{st.session_state.logger.get_record_count():,}")
        else:
            sim_data = [
                {"Parameter": "Simulation Engine", "Value": "Active" if st.session_state.sim_is_running else "Paused"},
                {"Parameter": "Target Update Interval", "Value": f"{st.session_state.sim_interval} seconds"},
                {"Parameter": "Output Dataset", "Value": "data/solar_dataset_demo.csv"},
                {"Parameter": "Data Provenance", "Value": "simulation_demo (isolated)"},
            ]
            st.dataframe(pd.DataFrame(sim_data), use_container_width=True)
            st.metric("Total Demo Records Generated", f"{st.session_state.demo_logger.get_record_count():,}")

    with col_h2:
        st.subheader("Active Safety Alerts")
        eval_record = lr if lr else {}
        alerts = evaluate_telemetry_alerts(eval_record)
        if alerts:
            for alt in alerts:
                if alt["level"] == "CRITICAL":
                    st.error(f"🚨 **{alt['title']}**: {alt['message']}")
                else:
                    st.warning(f"⚠️ **{alt['title']}**: {alt['message']}")
        else:
            st.success("✅ Zero active safety anomalies detected. System operating within normal thresholds.")


# =========================================================
# PAGE 9: PROJECT INFORMATION & DOCUMENTATION
# =========================================================
elif "9. Project Documentation" in selected_page:
    st.markdown('<div class="main-header">📘 Project Documentation & Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">BE CSE Final-Year Capstone Project Specifications</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Project Title:
        **AI-Based Solar Microgrid Management System Using IoT, Real-Time Sensor Data, CSV Dataset and Machine Learning**

        ---
        ### 1. Dual Operational Architecture
        This system implements two strictly isolated operational data streams:
        1. **Live Hardware Mode:**
           - Connects to an ESP32 board over USB Serial (115200 baud).
           - Reads DHT22 (GPIO 4), BH1750 (GPIO 21/22), and INA219 (GPIO 21/22).
           - Logs exclusively to `data/solar_dataset_live.csv` (`data_source=esp32_hardware`).
           - Rejects synthetic data and displays `DISCONNECTED — WAITING FOR REAL SENSOR DATA` when offline.
        2. **Simulation Demo Mode:**
           - Powered by `src/demo_data_generator.py`.
           - Generates smoothly fluctuating readings at configurable intervals (1s, 2s, 5s, 10s).
           - Logs exclusively to `data/solar_dataset_demo.csv` (`data_source=simulation_demo`).
           - Tagged everywhere with `SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS`.

        ---
        ### 2. Electrical Power Calculation
        $$\\text{Power (W)} = \\text{Voltage (V)} \\times \\text{Current (A)}$$
        $$\\text{Current (A)} = \\frac{\\text{Current (mA)}}{1000.0}$$
        $$\\text{Power (W)} = \\text{Bus Voltage (V)} \\times \\left(\\frac{\\text{Current (mA)}}{1000.0}\\right)$$
        """
    )

# ----------------- REFRESH LOOP -----------------
# Auto-refresh loop to make live stream and simulation appear live and responsive
if is_demo_mode and st.session_state.sim_is_running:
    time.sleep(float(st.session_state.sim_interval))
    st.rerun()
elif is_live_mode and st.session_state.serial_reader.is_connected:
    time.sleep(2.0)
    st.rerun()

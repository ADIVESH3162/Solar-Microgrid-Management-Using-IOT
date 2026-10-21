# AI-Based Solar Microgrid Management System

**A Complete Cyber-Physical IoT & Machine Learning Solar Energy Management System**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ESP32](https://img.shields.io/badge/Hardware-ESP32%20DevKit-red.svg)](https://www.espressif.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-brightgreen.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest%20Regressor-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-30%20Passing-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

The **AI-Based Solar Microgrid Management System** monitors real-time solar photovoltaic (PV) generation, forecasts short-term available solar power using machine learning, and dynamically evaluates whether generation is sufficient to power connected DC loads.

The system features **two distinct, strictly isolated operational modes**:
1. **🟢 Live Hardware Mode (Default):**
   - Connects to an ESP32 microcontroller board over USB serial at 115200 baud.
   - Reads physical sensor readings: DHT22 (GPIO4), BH1750 (GPIO21/22), INA219 (GPIO21/22).
   - Validates physical bounds and calculates power: $\text{Power (W)} = \text{Voltage (V)} \times (\text{Current (mA)} / 1000.0)$.
   - Logs exclusively to `data/solar_dataset_live.csv` with `data_source=esp32_hardware`.
   - When hardware is disconnected, displays `DISCONNECTED — WAITING FOR REAL SENSOR DATA` and shows unavailable values as `N/A`.
2. **🧪 Simulation Demo Mode:**
   - Powered by `src/demo_data_generator.py`.
   - Continuously generates smooth, realistic, physics-guided changing sensor readings at configurable intervals (1s, 2s, 5s, or 10s).
   - Values drift smoothly (simulating sunlight, clouds, and thermal inertia) rather than jumping wildly.
   - Power is calculated consistently from simulated voltage and current.
   - Logs exclusively to `data/solar_dataset_demo.csv` with `data_source=simulation_demo`.
   - Labeled everywhere with a prominent banner: `⚠️ SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS`.

---

## 2. Directory Structure

```
Solar_Microgrid_Management/
|-- app.py                              # Streamlit real-time interactive dashboard
|-- requirements.txt                    # Python dependencies
|-- README.md                           # Master user guide & documentation
|-- run_project.bat                     # One-click Windows project launcher
|-- train_model.bat                     # One-click Windows machine learning training script
|
|-- esp32_firmware/
|   |-- esp32_solar_sensors.ino         # Complete Arduino C++ sketch for ESP32
|   |-- libraries.txt                   # Arduino IDE library specifications & install commands
|   |-- wiring_guide.md                 # Detailed pinout matrix, breadboard wiring & circuit guide
|
|-- data/
|   |-- solar_dataset_live.csv          # Real hardware telemetry (data_source=esp32_hardware)
|   |-- solar_dataset_demo.csv          # Continuously changing simulation telemetry (data_source=simulation_demo)
|   |-- solar_dataset_sample.csv        # Realistic 24-hr synthetic benchmark sample dataset (1440 rows)
|   |-- solar_dataset_historical.csv    # 7-day multi-weather archive for ML training (2016 rows)
|   |-- data_dictionary.md              # Column specifications, physical units & ranges
|
|-- src/
|   |-- __init__.py                     # Package initialization
|   |-- config.py                       # Global constants, electrical thresholds, file paths
|   |-- demo_data_generator.py          # Real-time stateful demo simulator with smooth transitions
|   |-- sensor_serial.py                # Serial communication, port discovery & background worker
|   |-- csv_logger.py                   # Thread-safe CSV append engine with deduplication
|   |-- data_processor.py               # Energy integration (Wh), cleaning, rolling metrics
|   |-- feature_engineering.py          # Cyclical time encodings, lag features & split logic
|   |-- train_model.py                  # Random Forest training with chronological split
|   |-- predict_power.py                # Inference pipeline with physical constraint enforcement
|   |-- energy_management.py            # Microgrid decision engine (load matching, storage advice)
|   |-- alerts.py                       # Anomaly detection & electrical threshold monitoring
|
|-- models/
|   |-- solar_power_rf_model.joblib     # Trained Random Forest regression model
|   |-- model_metrics.json              # Model evaluation metrics & provenance metadata
|
|-- scripts/
|   |-- generate_sample_data.py         # Baseline sample dataset generator
|   |-- validate_dataset.py             # Rigorous data integrity & unit validator
|   |-- export_dataset.py               # Dataset filtering & export utility
|
|-- tests/
|   |-- test_csv_logger.py              # CSV file creation, locking & appending unit tests
|   |-- test_data_processor.py          # Energy integration & KPI extraction unit tests
|   |-- test_demo_data_generator.py     # Demo generator transitions, intervals, and CSV isolation tests
|   |-- test_energy_management.py       # Sufficiency & recommendation logic unit tests
|   |-- test_live_hardware_integrity.py # Live hardware integrity, rejection of fake data, deduplication
|   |-- test_sensor_serial.py           # Serial message parsing & bounds validation unit tests
|
|-- docs/                               # Engineering documentation guides & diagrams
```

---

## 3. Quickstart Guide (Windows)

### Option A: One-Click Automated Launcher
Double-click `run_project.bat` or run in terminal:
```cmd
.\run_project.bat
```
This automatically verifies Python, sets up dependencies, ensures models exist, and opens the dashboard at `http://localhost:8501`.

### Option B: Manual Terminal Execution
```powershell
# 1. Activate Python virtual environment (if using .venv)
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated tests (all 30 tests should pass)
pytest

# 4. Launch Streamlit Dashboard
streamlit run app.py
```

---

## 4. Operating the Dashboard

### 🟢 Live Hardware Mode (Default)
1. Plug your ESP32 board into the laptop using a micro-USB data cable.
2. In the sidebar under **ESP32 Hardware Connection**, select your COM port (or leave on `AUTO`).
3. Click **🔌 Connect**.
4. Genuine readings from your DHT22, BH1750, and INA219 sensors stream into the dashboard.
5. Measurements are continuously logged to `data/solar_dataset_live.csv`.
6. Click **📥 Download Live CSV** to export the collected measurements.
7. If the ESP32 is unplugged or disconnected, the dashboard displays:
   `⚠️ DISCONNECTED — WAITING FOR REAL SENSOR DATA`
   and shows unavailable metrics as `N/A`.

### 🧪 Simulation Demo Mode
1. In the sidebar under **System Operating Mode**, select **🧪 Simulation Demo Mode**.
2. The dashboard displays the banner:
   `⚠️ SIMULATED DATA — NOT REAL SENSOR MEASUREMENTS`
3. **Simulation Controls:**
   - **Update Interval:** Select `1`, `2`, `5`, or `10` seconds (default: 2s).
   - **Time Window:** Select `Last 20, 50, 100, 200, 500 readings` for time series graphs.
   - **Start / Stop:** Click **▶️ Start** to begin generating values, or **⏸️ Stop** to freeze generation.
   - **Download Demo CSV:** Click **📥 Download Demo CSV** to export `data/solar_dataset_demo.csv`.
   - **Clear Demo Data:** Expand **🗑️ Clear Demo Data**, check the confirmation checkbox, and click **Clear Demo Data Now**.
4. **6 Continuously Updating Plotly Graphs:**
   - Solar Power (W)
   - Solar Voltage (V)
   - Solar Current (mA)
   - Sunlight Intensity (lux)
   - Ambient Temperature (°C)
   - Relative Humidity (%)

---

## 5. ESP32 Hardware Wiring Guide

| Sensor | Sensor Pin | ESP32 Pin | Level | Function |
| :--- | :--- | :--- | :--- | :--- |
| **DHT22** | DATA (Pin 2) | **GPIO 4** | 3.3V Logic | Ambient temperature & relative humidity |
| **DHT22** | VCC & GND | 3.3V & GND | 3.3V DC | Power supply |
| **BH1750**| SDA | **GPIO 21** | 3.3V I2C | Ambient sunlight illuminance (0x23) |
| **BH1750**| SCL | **GPIO 22** | 3.3V I2C | I2C Clock |
| **BH1750**| VCC & GND | 3.3V & GND | 3.3V DC | Power supply |
| **INA219**| SDA | **GPIO 21** | 3.3V I2C | High-side current/voltage monitor (0x40) |
| **INA219**| SCL | **GPIO 22** | 3.3V I2C | I2C Clock |
| **INA219**| VIN+ | Solar Panel (+) | Up to 26V | High-side DC feed from solar panel |
| **INA219**| VIN- | DC Load (+) | Up to 26V | High-side DC output to microgrid load |
| **Circuit**| Panel / Load (-) | ESP32 GND | 0V | Common ground reference |

---

## 6. Electrical Power Calculation & INA219 Details

$$\text{Power (Watts)} = \text{Voltage (Volts)} \times \text{Current (Amperes)}$$
$$\text{Current (Amperes)} = \frac{\text{Current (mA)}}{1000.0}$$
$$\text{Power (W)} = \text{Bus Voltage (V)} \times \left(\frac{\text{Current (mA)}}{1000.0}\right)$$

- **Why Software Calculation?** The INA219 internal multiplier register (`getPower_mW()`) uses fixed-point divisor scaling with integer truncation. Directly computing $V \times I$ in software ensures mathematical exactness.
- **Circuit Placement:** The INA219 must be placed directly in the solar panel output circuit to measure photovoltaic power.

---

## 7. Automated Test Suite

Run all automated unit tests using pytest:
```powershell
pytest
```
All **30 tests** pass:
- `test_csv_logger.py`: Verifies CSV schema conformance and deduplication.
- `test_data_processor.py`: Verifies numerical energy integration ($Wh$) and KPI extraction.
- `test_demo_data_generator.py`: Verifies smooth transitions, power formula, interval pacing, and dataset isolation.
- `test_energy_management.py`: Verifies power sufficiency and dispatch recommendations.
- `test_live_hardware_integrity.py`: Verifies rejection of fake live readings, duplicate prevention, and disconnected hardware state.
- `test_sensor_serial.py`: Verifies JSON packet validation and physical bounds enforcement.

# Cliodynamics: Political Stress Index (PSI) for Germany

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/theory-Cliodynamics-orange.svg)](https://peterturchin.com/cliodynamics/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An advanced analytical framework applying Peter Turchin's Structural-Demographic Theory (SDT) to quantify and forecast political instability in modern Germany. This project digitizes complex socio-economic indicators into a unified Political Stress Index (PSI), identifying instability windows through 2027.

---

## Overview

This repository provides a complete data pipeline—from automated ingestion of German federal statistics (Destatis) to high-fidelity predictive modeling. By synthesizing variables like elite overproduction, wealth inequality, and state capacity, the model forecasts the structural pressures that lead to social instability or political re-alignments.

### Key Components of PSI v4:
- **Wealth Pump:** Tracking the flow of wealth from labor to elites via rent-to-wage ratios.
- **Elite Overproduction:** Comparing annual estimated graduates from 13 elite fields against available elite position openings. `frustrated_fraction = max(0, graduates − openings) / graduates`, where `graduates = enrolled × 0.70 / 5` and `openings = holders × 0.05`.
- **Studentflow:** Annual pipeline inflow of first-semester entrants (Destatis 21311-0012) — a leading indicator that predicts labour-market pressure ~5 years ahead.
- **State Capacity:** Measuring the state's ability to absorb stress through tax revenue stability and public sector staffing.
- **Youth Bulge:** Annual population share of ages 15–24 relative to the total population. `youth_bulge = (youth_pop / total_pop) / mean(youth_share)`.
- **Mass Mobilization:** Integrating strike data and real-time interest via Google Trends as proxies for social unrest.

---

## Theory and Methodology

The Political Stress Index is calculated using a multi-factor composite formula:

$$PSI = \text{rolling\_mean} \left( \frac{WealthPump \times ElitePressure \times MacroEcon \times FoodPump \times YouthBulge \times Strikes}{StateCapacity}, 12 \right)$$

*Note: All components are normalized/indexed to ensure statistical comparability.*

---

## Project Architecture

The pipeline follows a modular architecture for reliability and reproducibility:

```mermaid
graph LR
    A[Loaders] -->|Raw CSV/API| B[data/raw/]
    B --> C[Preprocessors]
    C -->|Processed Dataset| D[data/processed/]
    D --> E[Analysis/Viz]
    E -->|Artifacts| F[output/]
```

### Data Sources
- **Destatis (GENESIS API):** Wages, CPI, University enrollment (21311-0003) + Studentflow entrants (21311-0012), Demographics, Tax revenue, GDP, Public sector data.
- **WSI:** Strike records (Arbeitskampfbilanz).
- **Google Trends:** Real-time mobilization signals.
- **Mikrozensus:** Managerial employment data (Führungskräfte).

---

## Installation and Setup

### 1. Prerequisites
- Python 3.9 or higher
- Destatis GENESIS API credentials

### 2. Clone and Install
```bash
git clone https://github.com/your-username/Cliodynamics.git
cd Cliodynamics
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
DESTATIS_USER=your_username
DESTATIS_PASSWORD=your_password
```

---

## Project Execution

The project is designed to be run as a sequential pipeline:

1. **Data Ingestion:** Fetch raw data from federal APIs.
   ```bash
   python src/loaders/load_rent_and_wages.py
   python src/loaders/load_students.py           # enrolled stock (21311-0003)
   python src/loaders/load_studienanfaenger.py   # Studentflow inflow (21311-0012)
   python src/loaders/load_economic_indicators.py
   ```
2. **Preprocessing:** Clean, merge, and calculate the PSI.
   ```bash
   python src/preprocessors/process_base_wages.py
   python src/preprocessors/process_students.py
   python src/preprocessors/process_final_psi.py
   ```
3. **Visualization:** Generate dashboards.
   ```bash
   # Interactive HTML dashboard (opens in browser)
   python src/analysis/generate_dashboard.py

   # Full Streamlit dashboard (raw + processed views)
   streamlit run src/analysis/raw_data_dashboard.py
   ```

### Optional: Mobilization Data
```bash
python src/loaders/load_google_trends.py
python src/analysis/merge_trends.py
```

---

## Analysis Artifacts

The pipeline generates dashboards in the `output/` directory and via Streamlit:
- **`psi_v4_dashboard.html`**: Interactive Plotly dashboard — 6 panels: PSI, Mass Mobilization, Elite Pressure, Elite Pipeline Flow (Studentflow), Macro/Youth, State Capacity, Forecast to 2027.
- **`psi_v4_analysis.png`**: Static publication-ready visualization with the 2026–2027 instability forecast.
- **Streamlit app** (`raw_data_dashboard.py`): Full interactive UI with Raw Data view (12 source tabs) and Processed Data view (8 PSI component tabs).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Scientific Context

This work is an implementation of Cliodynamics, a transdisciplinary area of research that integrates historical macrosociology, cultural and social anthropology, and mathematical modeling of historical processes. It specifically utilizes the Structural-Demographic Theory (SDT) developed by Jack Goldstone and Peter Turchin.

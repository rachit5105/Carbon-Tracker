<!-- # A Machine Learning-powered Carbon Tracker project for CO₂ analysis.

# Carbon Tracker

Carbon Tracker is an initiative to analyze and visualize carbon dioxide (CO₂) emissions data for better environmental awareness and sustainability insights.

This project is part of the Edunet Internship Program and focuses on India’s carbon emission trends over the years.

---

## Project Overview

Carbon Tracker helps:
- Analyze CO₂ emissions by fuel type and industrial sector
- Track changes across years
- Estimate personal CO₂ impact using a CO₂ footprint calculator  

---

## Features

- Fuel-wise and sector-wise emissions analysis
- Year-over-year trend visualization
- Per capita emissions insights
- Clean and organized dataset structure
- Jupyter Notebook-based analysis

---

## Technologies Used

- Python 3
- Pandas
- Matplotlib / Seaborn
- Jupyter Notebook

---

## References

- International Energy Agency (IEA) – for CO₂ emissions-related insights and data references:  
  https://www.iea.org/countries/india/emissions

- Carbon Footprint Calculator – for emission estimation and contextual understanding:  
  https://www.carbonfootprint.com/calculator.aspx

- The dataset used in this project was generated with the assistance of ChatGPT by OpenAI, based on publicly available reference data formats.

---

## Developed By

- Rachit Patel  
- Pruthvi Thakor  
- Meet Patel  

> Developed as part of the Edunet Internship 2025 — Skill4Future Program.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/rachit5105/Carbon-Tracker.git
   cd Carbon-Tracker
   
2. Open the notebook using Jupyter:
   jupyter notebook main.ipynb
   
3. Ensure all dataset files are placed in the data source folder. -->



# 🌱 Carbon Tracker — Machine Learning-Powered CO₂ Analysis for India

A comprehensive machine learning-based project and web application for analyzing and visualizing CO₂ emissions in India, along with a personal carbon footprint calculator. Developed as part of the **Edunet Internship 2025 – Skill4Future Program**.

---

## 📌 Overview

**Carbon Tracker** aims to:
- Analyze CO₂ emissions by **fuel type**, **sector**, and **region**
- Track **historical emission trends** from 2000 to 2022
- Predict **individual carbon footprint** using lifestyle factors
- Provide **personalized suggestions** to reduce emissions

---

## ⚙️ Technologies Used

| Area | Tools |
|------|-------|
| Data Handling | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (Random Forest Regressor) |
| Web App | Streamlit |
| IDE | Jupyter Notebook, VS Code |

---

## 🚀 Features

### 📊 Emission Analysis Dashboard
- Fuel-wise & sector-wise breakdown (2000–2022)
- Per capita & total emission trends
- Interactive pie and bar charts
- India state-wise insights

### 🧮 Carbon Footprint Calculator
- Input 8 lifestyle factors: travel, food, electricity, etc.
- ML-powered real-time CO₂ prediction
- Personalized reduction suggestions
- Compare with national average

### 🤖 Machine Learning Insights
- Random Forest Regressor model
- Training, testing, and performance metrics
- Feature importance analysis

### 🎨 UI & UX
- Responsive design (Streamlit)
- Custom styling with gradients and emojis
- Fast and clean dashboard layout

---

## 🔧 How to Run

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py

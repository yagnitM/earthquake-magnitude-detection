# 🌍 Earthquake Magnitude Detection Based on Precursor Signals

A machine learning approach to predicting earthquake magnitudes using precursor seismic, geophysical, and environmental signals to enhance early warning systems.

---

## 📌 About This Project
Earthquakes pose significant threats to communities worldwide.  
This project explores **predicting earthquake magnitudes** by analyzing precursor signals that occur before seismic events. Using a dataset of approximately **7,500 earthquake events**, we built and compared several regression models to identify patterns that could improve disaster preparedness.

The goal is to contribute to **UN Sustainable Development Goal 3: Good Health and Well-being** by developing tools that could help save lives through better early warning systems.

---

## ⚡ Methodology

### 🔹 Data Processing
- Cleaned and preprocessed earthquake event data with multiple precursor signal features  
- Handled missing values and outliers in seismic measurements  
- Engineered temporal and spatial features for improved prediction  

### 🔹 Machine Learning Models
- **Linear Regression** – Baseline model for comparison  
- **Random Forest** – Ensemble method to capture non-linear relationships  
- **XGBoost** – Gradient boosting for higher accuracy  

### 🔹 Analytics & Visualization
- SQL queries for data aggregation and pattern discovery  
- Interactive **Tableau dashboard** featuring:
  - Geographic distribution of earthquake events  
  - Model prediction accuracy across regions  
  - Feature importance rankings  
  - Temporal trends in seismic activity  

---

## 📊 Results
- **Random Forest** and **XGBoost** outperformed Linear Regression with lower RMSE values  
- Key precursor signals (e.g., seismic frequency shifts, ground tilt) strongly influenced magnitude predictions  
- Tableau dashboard revealed **spatial clusters** of high-magnitude events and clear temporal patterns  

---

## 🛠️ Tech Stack
- **Python** – pandas, scikit-learn, xgboost, matplotlib  
- **SQL** – Data analysis & aggregation  
- **Tableau** – Interactive dashboards  
- **Jupyter Notebooks** – Development & experimentation  

---

## 🌱 SDG Alignment
This project aligns with **United Nations SDG 3: Good Health and Well-being**, aiming to reduce disaster risks and save lives through improved early prediction systems.

---

## 👨‍💻 Authors
- **Yagnit Mahajan**  
- **Ashwin Anand**

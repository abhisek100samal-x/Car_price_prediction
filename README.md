# 🚗 Ford Car Price Predictor

An interactive **Streamlit** web app that predicts the resale price of used Ford cars using **Linear Regression**, with full exploratory data analysis (EDA) and model-diagnostics dashboards built on **Plotly**.

> Upload a dataset → explore the data → understand the model → get an instant price estimate.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" alt="scikit-learn">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
</p>

---

## 📖 Overview

This project walks through a complete, end-to-end regression workflow — data cleaning, feature engineering, model training, evaluation, and deployment — wrapped in a polished, four-tab Streamlit interface:

| Tab | What it shows |
|---|---|
| 📊 **Data Overview** | Raw sample, dataset shape, missing-value check, descriptive statistics |
| 📈 **EDA Charts** | Interactive Plotly visualizations of price relationships and distributions |
| 🤖 **Model Insights** | Actual vs. predicted, residual diagnostics, and feature-coefficient impact |
| 💰 **Predict Price** | Live price estimate from user-entered vehicle attributes |

## ✨ Features

- **End-to-end ML pipeline** — cleaning, one-hot encoding, feature scaling, train/test split, and evaluation (R² and Adjusted R²) computed live from the uploaded CSV
- **Interactive EDA** — hover, zoom, and filter on every chart instead of static images
- **Model diagnostics** — actual-vs-predicted fit, residuals-vs-predicted (checks for heteroscedasticity), residual distribution, and MAE / RMSE
- **Feature impact** — top 15 features ranked by absolute regression coefficient, color-coded by direction of effect
- **Live prediction** — enter model, year, mileage, transmission, fuel type, tax, MPG, and engine size to get an instant estimated price
- **Cached training** — `st.cache_data` avoids retraining on every interaction
- **Custom UI theme** — clean navy-and-white design with metric cards, gradient hero banner, and styled tabs

## 🖼️ Visualizations included

**EDA tab**
- Price distribution (histogram + box marginal)
- Price vs. mileage (colored by fuel type)
- Price by transmission / fuel type (violin + box)
- Price vs. engine size
- Correlation of each numeric feature with price (sorted bar)
- Average price trend by year, with listing volume overlay
- Full correlation heatmap

**Model Insights tab**
- Actual vs. predicted scatter with a perfect-fit reference line
- Residuals vs. predicted (diagnostic for non-linearity/heteroscedasticity)
- Residual distribution
- Top-15 feature coefficients (magnitude and direction)

## 🧠 Model & Methodology

1. **Cleaning** — drop nulls, remove non-positive prices, negative mileage, and zero/negative engine sizes
2. **Encoding** — one-hot encode `model`, `transmission`, and `fuelType`
3. **Scaling** — `StandardScaler` applied to `year`, `mileage`, `tax`, `engineSize`
4. **Split** — 80/20 train-test split (`random_state=42`)
5. **Model** — `LinearRegression` (scikit-learn)
6. **Evaluation** — R², Adjusted R², MAE, RMSE, and residual analysis

> The app is dataset-driven: metrics and charts update automatically based on whatever CSV you upload, as long as it follows the expected schema (see below).

## 📂 Dataset

Built for the **Ford used-car listings** format (e.g., the "100,000 UK Used Car Data Set" on Kaggle), with columns:

```
model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize
```

No dataset is bundled with this repo — upload your own `ford.csv` from the app's sidebar. Not affiliated with Ford Motor Company; this project uses only publicly available used-car listing data.

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — app framework and UI
- **pandas / NumPy** — data manipulation
- **scikit-learn** — preprocessing and Linear Regression
- **Plotly** — interactive charts
- **Matplotlib / Seaborn** — supplementary static plotting

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/ford-car-price-predictor.git
cd ford-car-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use it
Open the local URL Streamlit prints (usually `http://localhost:8501`), upload your `ford.csv` in the sidebar, and explore the tabs.

## 📁 Project Structure

```
ford-car-price-predictor/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## ⚠️ Limitations

- Linear Regression assumes linear relationships; it will underfit non-linear price effects (e.g., steep depreciation curves)
- One-hot encoding is fit per session on the uploaded file, so model coverage depends on the categories present in that file
- No persistence — the trained model is not saved between sessions (retrains on every upload)

## 🔭 Future Improvements

- [ ] Add Ridge/Lasso/Random Forest/XGBoost as selectable models with side-by-side comparison
- [ ] Persist the trained model with `joblib` to avoid retraining on reload
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Support other car brands/datasets with configurable schemas
- [ ] Add SHAP-based explainability for individual predictions



## 👤 Author

**Abhisek**


Data Science Aspirant | Machine Learning & Full-Stack Enthusiast

If you found this useful, consider ⭐ starring the repo!

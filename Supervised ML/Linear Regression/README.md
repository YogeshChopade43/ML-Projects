Linear Regression (Salary prediction)
====================================

This small project trains and evaluates linear models (LinearRegression, RidgeCV, LassoCV) on a salary dataset.

Setup
-----
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run the training/evaluation script (default path expects `salary/Salary_dataset.csv`):

```powershell
# from project root
python lr_train.py --data salary/Salary_dataset.csv
```

Options
-------
- `--data` or `-d`: path to CSV or Excel file. Default: `salary/Salary_dataset.csv`.
- `--target` or `-t`: specify the target column name (default: last column).
- `--poly` or `-p`: polynomial degree (default 1). Use `--poly 2` to try degree-2 features.

Artifacts
---------
- `model.joblib` — saved best model pipeline and feature list.
- Plots — residuals are shown interactively when the script runs.

Next recommended steps
----------------------
1. Drop index-like columns before training (e.g. `Unnamed: 0`).
2. Try `--poly 2` to see if polynomial features help.
3. Compare tree-based models (RandomForest, LightGBM) for potential improvements.
4. If you plan to deploy, collect more data or use repeated CV / LOOCV to stabilize estimates.

If you'd like, I can automatically drop index columns and re-run experiments, add tree-based models for comparison, and output a short PDF report with metrics and plots.
# Salary Prediction using Linear Regression

This project implements a **Linear Regression model** to predict employee salaries based on their years of experience. It demonstrates the end-to-end workflow of a simple Machine Learning project — including data preprocessing, model training, evaluation, and visualization.

---

## 📂 Project Structure

```
├── salary/
│   └── Salary_dataset.csv     # Dataset
├── venv
├── lr_train.py
├── requirements.txt
├── README.md               # Project documentation
└── results/
    └── results.txt
```

---

## 🚀 Features

* Loads dataset of **Years of Experience vs Salary**
* Splits data into **train and test sets**
* Trains a **Linear Regression model** using scikit-learn
* Evaluates the model with **R² score and Mean Squared Error**
* Visualizes the regression line and predictions

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/salary-prediction.git
cd salary-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Place your dataset in the `data/` folder (default: `salary_data.csv`).
2. Run the training script:

```bash
python src/train_model.py
```

3. To generate a visualization of the regression line:

```bash
python src/visualize.py
```

---

## 📊 Example Results

* **Equation (approx):**

  ```
  Salary = 9300 * YearsExperience + 25000
  ```
* **R² Score:** 0.95
* **Visualization:**

![Regression Plot](results/regression_plot.png)

---

## 📚 Requirements

* Python 3.8+
* numpy
* pandas
* scikit-learn
* matplotlib

---

## 🔮 Future Work

* Extend model to support **multiple linear regression** (e.g., including education, role, location).
* Deploy the model as a **Flask API** for real-time predictions.
* Add **unit tests** for reproducibility.

---

## 📝 License

This project is licensed under the MIT License. Feel free to use and modify.

---

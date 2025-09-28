# Salary Prediction using Linear Regression

This project implements a **Linear Regression model** to predict employee salaries based on their years of experience. It demonstrates the end-to-end workflow of a simple Machine Learning project — including data preprocessing, model training, evaluation, and visualization.

---

## 📂 Project Structure

```
├── data/
│   └── salary_data.csv     # Dataset (YearsExperience, Salary)
├── src/
│   ├── train_model.py      # Train and evaluate linear regression
│   ├── visualize.py        # Plot regression line
│   └── utils.py            # Helper functions
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── results/
    └── regression_plot.png # Saved visualization
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

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


def prepare_xy(df, target_col=None):
    # Drop rows with missing values (simple approach)
    df = df.dropna()
    if target_col is None:
        # assume last column is target
        target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep numeric features only for this simple pipeline
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] != X.shape[1]:
        dropped = set(X.columns) - set(X_num.columns)
        print(f"Dropped non-numeric columns: {dropped}")
    return X_num, y


def evaluate_model(model, X_train, X_test, y_train, y_test, cv=5):
    # Train/test metrics
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cross-validated MSE (neg MSE returned)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_mse_scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    cv_r2_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kf)

    results = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'cv_mse_mean': float(np.mean(cv_mse_scores)),
        'cv_mse_std': float(np.std(cv_mse_scores)),
        'cv_r2_mean': float(np.mean(cv_r2_scores)),
        'cv_r2_std': float(np.std(cv_r2_scores)),
    }
    return results, y_pred


def print_results(name, res):
    print(f"\nModel: {name}")
    print(f"Test MSE: {res['mse']:.4f}")
    print(f"Test RMSE: {res['rmse']:.4f}")
    print(f"Test R^2: {res['r2']:.4f}")
    print(f"CV MSE: {res['cv_mse_mean']:.4f} ± {res['cv_mse_std']:.4f}")
    print(f"CV R^2: {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default='salary/Salary_dataset.csv', help='Path to CSV/XLSX data file')
    parser.add_argument('--target', '-t', default=None, help='Target column name (default: last column)')
    parser.add_argument('--poly', '-p', type=int, default=1, help='Polynomial degree (default 1 = linear)')
    args = parser.parse_args()

    df = load_data(args.data)
    print('Loaded data shape:', df.shape)
    X, y = prepare_xy(df, args.target)
    print('Using features:', list(X.columns))

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline: simple Linear Regression with scaling
    steps = []
    steps.append(('scaler', StandardScaler()))
    if args.poly and args.poly > 1:
        steps.append(('poly', PolynomialFeatures(degree=args.poly, include_bias=False)))
    steps.append(('lr', LinearRegression()))
    baseline = Pipeline(steps)

    base_res, base_pred = evaluate_model(baseline, X_train, X_test, y_train, y_test, cv=5)
    print_results('Baseline LinearRegression', base_res)

    # Ridge with built-in CV for alpha
    alphas = np.logspace(-3, 3, 25)
    steps_ridge = []
    steps_ridge.append(('scaler', StandardScaler()))
    if args.poly and args.poly > 1:
        steps_ridge.append(('poly', PolynomialFeatures(degree=args.poly, include_bias=False)))
    steps_ridge.append(('ridge', RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)))
    ridge_pipeline = Pipeline(steps_ridge)
    ridge_res, ridge_pred = evaluate_model(ridge_pipeline, X_train, X_test, y_train, y_test, cv=5)
    print_results('RidgeCV', ridge_res)

    # Lasso with CV (may be slower)
    lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(alphas=None, cv=5, max_iter=5000, n_alphas=50))
    ])
    try:
        lasso_res, lasso_pred = evaluate_model(lasso, X_train, X_test, y_train, y_test, cv=5)
        print_results('LassoCV', lasso_res)
    except Exception as e:
        print('LassoCV failed:', e)
        lasso_res = None

    # Pick best model by test RMSE
    candidates = [('Baseline', baseline, base_res), ('Ridge', ridge_pipeline, ridge_res)]
    if lasso_res is not None:
        candidates.append(('Lasso', lasso, lasso_res))

    best = min(candidates, key=lambda t: t[2]['rmse'])
    best_name, best_model, best_res = best
    print(f"\nBest model by test RMSE: {best_name} (RMSE={best_res['rmse']:.4f})")

    # Fit best model on full training data and save
    best_model.fit(X_train, y_train)
    model_path = 'model.joblib'
    joblib.dump({'model': best_model, 'features': list(X.columns)}, model_path)
    print(f"Saved best model to {model_path}")

    # Residual plot for best model
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals distribution')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
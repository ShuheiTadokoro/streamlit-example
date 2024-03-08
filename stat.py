import os

if "PYTHONPATH" in os.environ:
    print(os.environ["PYTHONPATH"])
else:
    print("PYTHONPATH環境変数が存在しません。")
    os.environ["PYTHONPATH"] = "."
import subprocess

def install_package(package):
    subprocess.check_call(["pip", "install", package])

# 必要なライブラリのインストール
# install_package("pmdarima")
# install_package("pykalman")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from statsmodels.graphics.tsaplots import plot_acf

def perform_linear_regression(data):
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    coefficients = model.coef_
    coefficients_with_intercept = np.insert(coefficients, 0, model.intercept_)

    residuals = y - y_pred
    residual_std = np.std(residuals, ddof=X.shape[1])

    mse = ((y - y_pred) ** 2).sum() / (len(y) - X.shape[1] - 1)
    se = np.sqrt(np.diag(mse * np.linalg.inv(np.dot(X.T, X))))
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
    var_beta = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
    se_with_intercept = np.sqrt(np.diag(var_beta))
    t_values = coefficients_with_intercept / se_with_intercept
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=len(y) - X.shape[1] - 1))
    statistics = pd.DataFrame({"係数": coefficients_with_intercept, "標準誤差": se_with_intercept, "t値": t_values, "p値": p_values}, index=["切片"] + list(X.columns))

    return model, r2, statistics, y, y_pred

def perform_logistic_regression(data):
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    model = LogisticRegression()
    model.fit(X, y)

    odds_ratio = np.exp(model.coef_.ravel())
    n = len(y)
    p = X.shape[1]
    p_values = 2 * (1 - stats.norm.cdf(np.abs(model.coef_.ravel())))
    statistics = pd.DataFrame({"係数": model.coef_.ravel(), "p値": p_values, "オッズ比": odds_ratio}, index=X.columns)
    statistics.loc["切片"] = [model.intercept_[0], np.nan, np.nan]

    return model, statistics, X, y

def perform_arima(data, date_column):
    if date_column not in data.columns:
        raise ValueError(f"Column '{date_column}' not found in the dataframe.")

    dates = pd.to_datetime(data[date_column])
    data_numeric = data.drop(columns=[date_column])

    model = auto_arima(data_numeric, seasonal=True, m=12)
    forecast = model.predict(n_periods=len(dates))

    plot_arima_diagnostic_plots(model)  # 新しく追加した関数を呼び出す

    return model, forecast, dates


def plot_arima_diagnostic_plots(model):
    # Standardized Residuals
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    model.plot_diagnostics(fig=fig)
    plt.tight_layout()
    st.pyplot()

    # Histogram plus estimated density
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.hist(model.resid(), bins=20, density=True, alpha=0.6, color='b')
    ax.set_title('Histogram plus estimated density')
    ax.set_xlabel('Residuals')
    st.pyplot()

    # Normal Q-Q plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    stats.probplot(model.resid(), dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot')
    st.pyplot()

    # Correlogram
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plot_acf(model.resid(), lags=20, ax=ax)
    ax.set_title('Correlogram')
    st.pyplot()

def perform_kalman_filter(data, date_column):
    dates = pd.to_datetime(data[date_column])
    data_numeric = data.drop(columns=[date_column])

    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    fitted_model = kf.em(data_numeric).smooth(data_numeric)
    forecast = fitted_model[0][-len(dates):]

    return fitted_model, forecast, dates

def plot_forecast_with_dates(forecast, dates):
    plt.plot(dates, forecast)
    plt.xlabel('Date')
    plt.ylabel('Forecast')
    plt.title('Forecast with Dates')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot()


def perform_clustering(data, algorithm='kmeans'):
    X = data.iloc[:, 1:]  # 1列目以降を説明変数として抽出
    
    if algorithm == 'kmeans':
        # KMeansクラスタリングの実行
        model = KMeans(n_clusters=3, random_state=42)
    elif algorithm == 'hierarchical':
        # 階層的クラスタリングの実行
        model = AgglomerativeClustering(n_clusters=3)
    else:
        raise ValueError("Invalid clustering algorithm selected")

    cluster_labels = model.fit_predict(X)

    return cluster_labels

def plot_cluster_bar_chart(data, cluster_labels):
    cluster_data = pd.concat([data, pd.DataFrame({'cluster_id': cluster_labels})], axis=1)
    clusterinfo = pd.DataFrame()
    for i in range(3):  # クラスタ数に合わせて修正
        clusterinfo['cluster' + str(i)] = cluster_data[cluster_data['cluster_id'] == i].mean()
    clusterinfo = clusterinfo.drop('cluster_id')

    # 可視化（積み上げ棒グラフ）
    my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of Clusters")
    my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

    # 数値を表示
    for p in my_plot.patches:
        my_plot.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005))

    st.pyplot()

def main():
    st.title("統計分析アプリ")

    model_option = st.sidebar.selectbox("モデルを選択してください", ["重回帰分析", "ロジスティック回帰", "ARIMA", "カルマンフィルタ","非階層クラスタリング", "階層クラスタリング"])

    if model_option in ["重回帰分析", "ロジスティック回帰", "ARIMA", "カルマンフィルタ","非階層クラスタリング", "階層クラスタリング"]:
        st.subheader("ファイルをアップロードしてください")
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください,ARIMAとカルマンフィルタの場合は日付カラム名は'Month'としてください。", type=['csv'])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            st.subheader("基本統計量")
            basic_statistics = data.iloc[:, 1:].describe()
            st.write(basic_statistics)

            st.subheader("データ量")
            st.write("行数:", data.shape[0])
            st.write("列数:", data.shape[1])

            st.subheader("平均")
            st.write(data.iloc[:, 1:].mean())

            st.subheader("標準偏差")
            st.write(data.iloc[:, 1:].std())

            st.subheader("相関行列")
            correlation_matrix = data.iloc[:, 1:].corr()
            st.write(correlation_matrix)

            st.subheader("分散共分散行列")
            covariance_matrix = data.iloc[:, 1:].cov()
            st.write(covariance_matrix)

            if model_option == "重回帰分析":
                st.subheader("重回帰分析の結果")
                model, r2, statistics, y, y_pred = perform_linear_regression(data)
                st.write("決定係数 (R2乗):", r2)
                st.write("統計量:")
                st.write(statistics)
                plt.scatter(y, y_pred)
                plt.xlabel("実績")
                plt.ylabel("予測値")
                st.pyplot()

            elif model_option == "ロジスティック回帰":
                st.subheader("ロジスティック回帰の結果")
                model, statistics, X, y = perform_logistic_regression(data)
                st.write("統計量:")
                st.write(statistics)
                plt.scatter(y, model.predict(X))
                plt.xlabel("実績")
                plt.ylabel("予測値")
                st.pyplot()

            elif model_option == "ARIMA":
                st.subheader("ARIMAの結果")
                model, forecast, dates = perform_arima(data, "Month")
                st.write("予測結果:")
                st.write(forecast)
                plot_forecast_with_dates(forecast, dates)

            elif model_option == "カルマンフィルタ":
                st.subheader("カルマンフィルタの結果")
                fitted_model, forecast, dates = perform_kalman_filter(data, "Month")
                st.write("予測結果:")
                st.write(forecast)
                plot_forecast_with_dates(forecast, dates)

            elif model_option == "非階層クラスタリング":
                st.subheader("非階層クラスタリングの結果")
                cluster_labels = perform_clustering(data, algorithm='kmeans')
                plot_cluster_bar_chart(data, cluster_labels)

            elif model_option == "階層クラスタリング":
                st.subheader("階層クラスタリングの結果")
                cluster_labels = perform_clustering(data, algorithm='hierarchical')
                plot_cluster_bar_chart(data, cluster_labels)
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import scipy.stats as stats

def perform_linear_regression(data):
    X = data.iloc[:, 1:]  # 1列目以降を説明変数として抽出
    y = data.iloc[:, 0]   # 0列目を被説明変数として抽出

    model = LinearRegression()
    model.fit(X, y)

    # 決定係数(R2乗)を計算
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # 標準誤差を計算
    n = len(y)
    p = X.shape[1]
    mse = ((y - y_pred) ** 2).sum() / (n - p - 1)
    se = np.sqrt(np.diag(mse * np.linalg.inv(np.dot(X.T, X))))

    # t値を計算
    t_values = model.coef_ / se

    # p値を計算
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=n-p-1))

    # 統計量のDataFrameを作成
    statistics = pd.DataFrame({"係数": model.coef_, "標準誤差": se, "t値": t_values, "p値": p_values}, index=X.columns)
    statistics.loc["切片"] = [model.intercept_, np.nan, np.nan, np.nan]

    return model, r2, statistics

def perform_logistic_regression(data):
    X = data.iloc[:, 1:]  # 1列目以降を説明変数として抽出
    y = data.iloc[:, 0]   # 0列目を被説明変数として抽出

    model = LogisticRegression()
    model.fit(X, y)

    # オッズ比を計算
    odds_ratio = np.exp(model.coef_[0])

    # p値を計算
    n = len(y)
    p = X.shape[1]
    p_values = 2 * (1 - stats.norm.cdf(np.abs(model.coef_)))

    # 統計量のDataFrameを作成
    statistics = pd.DataFrame({"係数": model.coef_[0], "p値": p_values, "オッズ比": odds_ratio}, index=X.columns)
    statistics.loc["切片"] = [model.intercept_[0], np.nan, np.nan]

    return model, statistics

def perform_clustering(data):
    X = data.iloc[:, 1:]  # 1列目以降を説明変数として抽出

    # クラスタリングの実行
    model = KMeans(n_clusters=3)  # クラスタ数を3として設定
    model.fit(X)

    cluster_labels = model.labels_
    return cluster_labels

def plot_scatter(x, y, y_pred, model_name=""):
    plt.scatter(x, y, label="実績値")
    plt.plot(x, y_pred, color='red', label="予測値")
    plt.xlabel("説明変数")
    plt.ylabel("被説明変数")
    plt.title(f"{model_name}の散布図と回帰直線")
    plt.legend()
    st.pyplot()

def main():
    st.title("統計分析アプリ")

    # CSVファイルのアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        columns = data.columns.tolist()

        # 基本統計量を表示
        st.subheader("基本統計量")
        st.write(data.describe())

        # 平均を表示
        st.subheader("平均")
        st.write(data.mean())

        # 相関行列を表示
        st.subheader("相関行列")
        st.write(data.corr())

        # 標準偏差を表示
        st.subheader("標準偏差")
        st.write(data.std())

        # モデルの選択
        model_option = st.selectbox("モデルを選択してください", ["重回帰分析", "ロジスティック回帰", "クラスタリング"])

        if model_option == "重回帰分析":
            # 重回帰分析の実行
            st.subheader("重回帰分析の結果")
            model, r2, statistics = perform_linear_regression(data)
            st.write("決定係数 (R2乗):", r2)
            st.write("統計量:")
            st.write(statistics)

            # 散布図のプロット
            predicted_values = model.predict(data.iloc[:, 1:])
            plot_scatter(data.iloc[:, 0], data.iloc[:, 0], predicted_values, model_name="重回帰分析")

            # 回帰式の表示
            coefficients = model.coef_
            intercept = model.intercept_
            st.write(f"回帰式: Y = {intercept} + ", end="")
            for i, col in enumerate(data.columns[1:]):
                st.write(f"{coefficients[i]}*{col}", end="")
                if i < len(coefficients) - 1:
                    st.write(" + ", end="")
            st.write("")

        elif model_option == "ロジスティック回帰":
            # ロジスティック回帰の実行
            st.subheader("ロジスティック回帰の結果")
            model, statistics = perform_logistic_regression(data)
            st.write("統計量:")
            st.write(statistics)

        elif model_option == "クラスタリング":
            # クラスタリングの実行
            st.subheader("クラスタリングの結果")
            cluster_labels = perform_clustering(data)
            st.write("クラスタリング結果:", cluster_labels.tolist())

if __name__ == "__main__":
    main()

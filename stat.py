import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
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
    odds_ratio = np.exp(model.coef_.ravel())

    # p値を計算
    n = len(y)
    p = X.shape[1]
    p_values = 2 * (1 - stats.norm.cdf(np.abs(model.coef_.ravel())))

    # 統計量のDataFrameを作成
    statistics = pd.DataFrame({"係数": model.coef_.ravel(), "p値": p_values, "オッズ比": odds_ratio}, index=X.columns)
    statistics.loc["切片"] = [model.intercept_[0], np.nan, np.nan]

    return model, statistics

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
    st.pyplot()

def main():
    st.title("統計分析アプリ")

    # モデルの選択
    model_option = st.sidebar.selectbox("モデルを選択してください", ["重回帰分析", "ロジスティック回帰", "非階層クラスタリング", "階層クラスタリング"])

    if model_option != "ファイルアップロード":
        # CSVファイルのアップロード
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            if model_option == "重回帰分析":
                # 重回帰分析の実行
                st.subheader("重回帰分析の結果")
                model, r2, statistics = perform_linear_regression(data)
                st.write("決定係数 (R2乗):", r2)
                st.write("統計量:")
                st.write(statistics)

                # 基本統計量の表示
                st.subheader("基本統計量")
                st.write("データ数:", len(data))
                st.write("平均:")
                st.write(data.mean())
                st.write("標準偏差:")
                st.write(data.std())
                st.write("相関係数行列:")
                st.write(data.corr())
                st.write("分散共分散行列:")
                st.write(data.cov())

            elif model_option == "ロジスティック回帰":
                # ロジスティック回帰の実行
                st.subheader("ロジスティック回帰の結果")
                model, statistics = perform_logistic_regression(data)
                st.write("統計量:")
                st.write(statistics)

                # 基本統計量の表示
                st.subheader("基本統計量")
                st.write("データ数:", len(data))
                st.write("平均:")
                st.write(data.mean())
                st.write("標準偏差:")
                st.write(data.std())
                st.write("相関係数行列:")
                st.write(data.corr())
                st.write("分散共分散行列:")
                st.write(data.cov())

            elif model_option == "非階層クラスタリング":
                # 非階層的クラスタリングの実行
                st.subheader("非階層クラスタリングの結果")
                cluster_labels = perform_clustering(data, algorithm='kmeans')
                plot_cluster_bar_chart(data, cluster_labels)

            elif model_option == "階層クラスタリング":
                # 階層的クラスタリングの実行
                st.subheader("階層クラスタリングの結果")
                cluster_labels = perform_clustering(data, algorithm='hierarchical')
                plot_cluster_bar_chart(data, cluster_labels)

    else:
        st.write("ファイルアップロード前にモデルを選択してください。")

if __name__ == "__main__":
    main()

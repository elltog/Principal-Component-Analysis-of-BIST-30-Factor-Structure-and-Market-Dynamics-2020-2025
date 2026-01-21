import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis import run_pca
import yfinance as yf

def plot_corr(corr):
    plt.figure(figsize=(14,10))
    sns.heatmap(corr,annot=True,fmt='.2f', cmap="RdYlGn",center=0,square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix")
    plt.show()

def plot_pca(data):
    
    pca = PCA()
    result = pca.fit_transform(data)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

    i=0
    for ratio in np.cumsum(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.3f}")
        i+=1

def plot_pca2(data):

    pca = PCA()
    result = pca.fit_transform(data)

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(data.columns))], index=data.columns)

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(loadings.index, loadings[f'PC{i+1}'], marker='o', linewidth=2, markersize=8, label=f'PC{i+1}')

    plt.xlabel('Stocks')
    plt.ylabel('Loading Value')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pca3(data):
    window_size = 120
    rolling_pca = []

    for i in range(window_size, len(data)):
        window_data = data.iloc[i-window_size:i]
        window_scaled = StandardScaler().fit_transform(window_data)
        pca_window = PCA()
        pca_window.fit(window_scaled)
        rolling_pca.append(pca_window.explained_variance_ratio_[:5])

    rolling_pca_df = pd.DataFrame(rolling_pca, columns=['PCA 1', 'PCA 2', 'PCA 3','PCA 4', 'PCA 5'])

    plt.figure(figsize=(10, 5))
    for col in rolling_pca_df.columns:
        plt.plot(rolling_pca_df.index, rolling_pca_df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Rolling Explained Variance Ratio (60-day window)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pca4(data):

    window_size = 60
    rolling_pca = []

    for i in range(window_size, len(data)):
        window_data = data.iloc[i-window_size:i]
        window_scaled = StandardScaler().fit_transform(window_data)
        pca_window = PCA()
        pca_window.fit(window_scaled)
        rolling_pca.append(pca_window.explained_variance_ratio_[1:5])

    rolling_pca_df = pd.DataFrame(rolling_pca, columns=['PCA 2','PCA 3','PCA 4','PCA 5'])

    # Plot rolling explained variance
    plt.figure(figsize=(10, 5))
    for col in rolling_pca_df.columns:
        plt.plot(rolling_pca_df.index, rolling_pca_df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Rolling Explained Variance Ratio (60-day window)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pca5(data):
  
    bist30_price = yf.download("XU030.IS", start=data.index.min(), end=data.index.max(), progress=False)["Close"].dropna()
    bist30_ret = np.log(bist30_price / bist30_price.shift(1)).dropna()

    scaler = StandardScaler()
    Z = scaler.fit_transform(data)

    pca = PCA()
    pca.fit(Z)

    w_pc1 = pd.Series(pca.components_[0], index=data.columns)
    w_pc1 = w_pc1 / np.sum(np.abs(w_pc1))
    pc1_ret = data @ w_pc1

    w_pc2 = pd.Series(pca.components_[1], index=data.columns)
    w_pc2 = w_pc2 / np.sum(np.abs(w_pc2))
    pc2_ret = data @ w_pc2

    common_index = pc1_ret.index.intersection(bist30_ret.index)
    pc1_ret = pc1_ret.loc[common_index]
    pc2_ret = pc2_ret.loc[common_index]
    bist30_ret = bist30_ret.loc[common_index]

    plt.figure(figsize=(10,4))
    plt.plot((1 + pc1_ret).cumprod(), label="PC1 (Market Factor)")
    plt.plot((1 + pc2_ret).cumprod(), label="PC2 (Market-Neutral Factor)")
    plt.plot((1 + bist30_ret).cumprod(), label="BIST30 Index")
    plt.legend()
    plt.title("PC1 & PC2 Eigen-Portfolios vs BIST30")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

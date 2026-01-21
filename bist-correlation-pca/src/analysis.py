import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def correlation(data):
    cor1 = data.corr()[data.corr() !=1].max().sort_values()
    cor2 = data.corr()
    return cor1, cor2

def avrg_correlation(cor2):
    avg_correlation = (cor2.sum().sum() - len(cor2)) / (len(cor2) * (len(cor2) - 1))
    return avg_correlation

def run_pca(data):
    sc = StandardScaler()
    returns_scaled = sc.fit_transform(data)

    pca = PCA()
    pca_result = pca.fit_transform(returns_scaled)

    return pca

def regression(data):

    bist30_price = yf.download("XU030.IS", start=data.index.min(), end=data.index.max(), progress=False)["Close"].dropna()
    bist30_ret = np.log(bist30_price / bist30_price.shift(1)).dropna()

    scaler = StandardScaler()
    Z = scaler.fit_transform(data)

    pca = PCA()
    pca.fit(Z)

    w_pc1 = pd.Series(pca.components_[0], index=data.columns)
    w_pc1 = w_pc1 / np.sum(np.abs(w_pc1))
    pc1_ret = data @ w_pc1

    common_index = pc1_ret.index.intersection(bist30_ret.index)
    pc1_ret = pc1_ret.loc[common_index]
    bist30_ret = bist30_ret.loc[common_index]

    common_idx = pc1_ret.index.intersection(bist30_ret.index)

    Y = bist30_ret.loc[common_idx]
    X = pc1_ret.loc[common_idx]

    X = sm.add_constant(X)

    model_market = sm.OLS(Y, X).fit()
    return model_market


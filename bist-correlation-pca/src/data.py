import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def data():
    tickers = ["AKBNK.IS","ALARK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS","HEKTS.IS","ISCTR.IS","KCHOL.IS","KOZAL.IS","KRDMD.IS","ODAS.IS","PETKM.IS","SAHOL.IS","SASA.IS","SISE.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TOASO.IS","TUPRS.IS","VAKBN.IS","YKBNK.IS","ZOREN.IS","ARCLK.IS"]

    df = yf.download(tickers, start="2020-01-01", end="2025-01-31", auto_adjust=True, progress=False)["Close"]
    df = df.rename(columns={old: old[:-3] for old in df.columns})    
    
    df = df.ffill()
    df = df.bfill()
    rows=df.isnull().sum()[df.isnull().sum() > 1].index
    df.drop(columns=rows, inplace=True)
    print(f"{rows}")

    log = np.log(df / df.shift(1)) 
    log.drop(df.index[0], inplace=True)

    return df, log
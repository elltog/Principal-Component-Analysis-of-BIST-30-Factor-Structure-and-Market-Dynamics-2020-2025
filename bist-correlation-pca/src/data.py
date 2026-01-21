import yfinance as yf
import numpy as np

def data():
    tickers = ["AKBNK.IS","ALARK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS","HEKTS.IS","ISCTR.IS","KCHOL.IS","KOZAL.IS","KRDMD.IS","ODAS.IS","PETKM.IS","SAHOL.IS","SASA.IS","SISE.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TOASO.IS","TUPRS.IS","VAKBN.IS","YKBNK.IS","ZOREN.IS","ARCLK.IS"]

    df = yf.download(tickers, start="2020-01-01", end="2025-01-31")["Close"]
    df = df.rename(columns={old: old[:-3] for old in df.columns})

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    print("\n=== Missing Data Summary ===")
    print(f"Total observations per stock: {len(df)}")
    print(f"\nStocks with missing data:")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))
    
    # %10'dan fazla eksik verisi olan hisseleri çıkar
    threshold_pct = 10
    rows_to_drop = missing_pct[missing_pct > threshold_pct].index
    
    if len(rows_to_drop) > 0:
        print(f"\n⚠️  Dropping stocks with >{threshold_pct}% missing data: {list(rows_to_drop)}")
        df = df.drop(columns=rows_to_drop)
    
    # Kalan eksik verileri forward-fill ile doldur (güncellenmiş syntax)
    df = df.ffill()
    
    # Hala eksik veri varsa (örneğin ilk satırlar) backward-fill yap
    df = df.bfill()

    # rows=df.isnull().sum()[df.isnull().sum() > 1].index
    # df.drop(columns=rows, inplace=True)
    # print(f"{rows}")

    df = df.fillna(method="ffill")

    log = np.log(df / df.shift(1)) 
    log.drop(df.index[0], inplace=True)

    return df, log
import pandas as pd

def load_sample_data():
    df = pd.read_csv("data/bias_data.csv")
    return df
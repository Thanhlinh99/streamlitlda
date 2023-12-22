import pandas as pd
from prophet import Prophet
import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np

class ProphetModel:
    def __init__(self, changepoint_prior_scale=0.01):
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    
    def fit(self, df):
        self.model.fit(df[['datetime', 'PO4T1']].rename(columns={'datetime': 'ds', 'PO4T1': 'y'}))
        # self.model.fit(df[['datetime', 'MKN']].rename(columns={'datetime': 'ds', 'MKN': 'y'}))
        # self.model.fit(df[['datetime', 'Oxi']].rename(columns={'datetime': 'ds', 'Oxi': 'y'}))
    
    def make_future_dataframe(self, periods, freq):
        custom_freq = pd.DateOffset(seconds=freq)
        future = self.model.make_future_dataframe(periods, freq=custom_freq, include_history=False)
        return future
    
    def predict(self, future):
        return self.model.predict(future)
    
    def plot(self, fcst):
        fig = self.model.plot(fcst)
        return fig
    
    def plot_components(self, fcst):
        fig = self.model.plot_components(fcst)
        return fig
    
    def predict_values(self, fcst):
        return fcst[['ds','yhat']]
    
    def accuracy(self, predict_values, data_subset):
        acc = []
        for i in range(len(predict_values)):
             acc.append(abs(predict_values.iloc[i]['yhat'] - data_subset.iloc[i]) * 100 / predict_values.iloc[i]['yhat'])
        return np.average(np.array(acc))
    
def main():
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('numexpr').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    df = pd.read_csv('/home/linhtt/Documents/ProjectLDA/processingDataLDA/data_train.csv')
    df1 = pd.read_csv('/home/linhtt/Documents/ProjectLDA/processingDataLDA/data_validate.csv')

    prophet_model = ProphetModel(changepoint_prior_scale=0.01)
    prophet_model.fit(df)

    future = prophet_model.make_future_dataframe(periods=96, freq=7200)
    fcst = prophet_model.predict(future)
    
    prophet_model.plot(fcst)
    prophet_model.plot_components(fcst)
    data_subset = df1.loc[6912:7009, 'PO4T1'].astype(float)
    # data_subset = df1.loc[6912:7009, 'datetime']
    print(data_subset)

    predict_values = prophet_model.predict_values(fcst)

    predict_values_ds = predict_values[['ds', 'yhat']]
    print(predict_values_ds)

    acc = prophet_model.accuracy(predict_values, data_subset)
    print(f"PO4T1: {100-acc:.2f}%")

    plt.show()
    return predict_values_ds 
if __name__ == "__main__":
    main()

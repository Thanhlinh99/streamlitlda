from prophet import Prophet
import pandas as pd
import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('numexpr').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class TimeSeriesPredictor:
    def __init__(self, train_data_path, validate_data_path):
        # Load data train và data validate
        self.df_train = pd.read_csv(train_data_path)
        self.df_validate = pd.read_csv(validate_data_path)
        self.model = None
        self.future = None
        self.fcst = None
        # print(self.df_validate.loc[2037:2135, 'datetime'])

    # Những giá trị nào bị vượt ngưỡng thì random lại nó nằm trong ngưỡng an toàn
    def fix_data_over_value(self, column_name, lower_bound, upper_bound):
        data = self.df_train[column_name].values
        for i in range(len(data)):
            if lower_bound <= data[i] <= upper_bound:
                pass
            else:
                # data[i] = (lower_bound + upper_bound) / 2
                data[i] = np.random.uniform(lower_bound,upper_bound)

    # Train model dự đoán với target column là cột cần dự đoán
    def train_model(self, target_column, changepoint_prior_scale=0.1):
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale).fit(
            self.df_train[['datetime', target_column]].rename(columns={'datetime': 'ds', target_column: 'y'})
        )

    # Tạo dataframe predict để dự đoán giá trị với 
    # num_points là số điểm cần dự đoán
    # custom_fre là tầm suất của data predict
    def create_future_dataframe(self, num_points, custom_freq):
        self.future = self.model.make_future_dataframe(num_points, freq=custom_freq, include_history=False)

    # Predict data 
    def predict(self):
        self.fcst = self.model.predict(self.future)

    # Trực quan hóa kết quả dự đoán 
    def visualize(self):
        fig = self.model.plot(self.fcst)
    
    # Tính độ chính xác của model, dựa vào data thật và data dự đoán
    def calculate_accuracy(self, data_subset):
        predicted_values = self.fcst['yhat']
        acc = []

        for i in range(len(predicted_values)):
            acc.append(abs(predicted_values[i] - data_subset.iloc[i]) * 100 / predicted_values[i])

        accuracy = 100 - np.average(np.array(acc))
        return accuracy


def main():
    
    def process_predictor(predictor, target_column, lower_bound=None, upper_bound=None):
        if lower_bound is not None and upper_bound is not None:
            predictor.fix_data_over_value(column_name=target_column, lower_bound=lower_bound, upper_bound=upper_bound)
        predictor.train_model(target_column)
        predictor.create_future_dataframe(96, custom_freq='2H')
        predictor.predict()
        predictor.visualize()
        accuracy = predictor.calculate_accuracy(predictor.df_validate.loc[2037:2135, target_column].astype(float))
        predict_values = predictor.fcst[['ds', 'yhat']]
        # print(predict_values['ds'])
        print(accuracy)
        return predict_values, accuracy

    path_to_data_train = 'data_train1.csv'
    path_to_data_validate = 'data_validate1.csv'
    mkn_predictor = TimeSeriesPredictor(path_to_data_train, path_to_data_validate)
    mkn_values, accuracy_mkn = process_predictor(mkn_predictor, 'MKN')

    PO4T1_predictor = TimeSeriesPredictor(path_to_data_train, path_to_data_validate)
    PO4T1_values, accuracy_PO4T1 = process_predictor(PO4T1_predictor, 'PO4T1', lower_bound=900, upper_bound=1000)

    idtocdoquat_predictor = TimeSeriesPredictor(path_to_data_train, path_to_data_validate)
    idtocdoquat_values, accuracy_idtocdoquat = process_predictor(idtocdoquat_predictor, 'ID toc do quat')

    plt.show()

    return {
        'mkn': mkn_values,
        'PO4T1': PO4T1_values,
        'idtocdoquat': idtocdoquat_values,
        'accuracy_mkn': accuracy_mkn,
        'accuracy_PO4T1': accuracy_PO4T1,
        'accuracy_idtocdoquat': accuracy_idtocdoquat
    }

if __name__ == "__main__":
    main()

import streamlit as st
from streamlit_echarts import st_echarts
from pyecharts import options as opts
from pyecharts.charts import Line
import pandas as pd
import codecs
import plotly.express as px
import plotly.graph_objects as go
from trainModelProphet import main
import numpy as np
import pickle

st.set_page_config(page_title="Dashboard", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
def loadCSV():
    # Đọc dữ liệu từ tập tin CSV train
    df = pd.read_csv("/home/linhtt/Documents/ProjectLDA/processingDataLDA/data_train1.csv")
    # Đọc dữ liệu từ tập tin CSV add thêm để kiểm tra độ chính xác
    df1 = pd.read_csv('/home/linhtt/Documents/ProjectLDA/processingDataLDA/data_add_streamlit.csv')
    return df, df1

# đọc giá trị cuối cùng / giá trị hiện tại của csv để lấy giá trị hiện tại hiển thị lên đồng hồ
def readLastCSV(df): 
    return df['PO4T1'].iloc[-1], df['MKN'].iloc[-1], df['ID toc do quat'].iloc[-1]

#tạo giao diện chính 
def main_dashboard():
    # Thêm CSS và hình nền
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown("""
        <style>
            .rectangle1 {
                height: 130px;
                width: 100%;
                background-color: #06183B;
                position: fixed;
                top: 0px;
                left: 0px;
                z-index: 1;
            }
            .rectangle2 {
                height: 100%;
                width: 80px;
                background-color: #06183B;
                position: fixed;
                top: 0px;
                left: 0px;
                z-index: 2;
            }
            .backgroundLDA {
                position: fixed;
                top: 130px;
                left: 80px;
                z-index: 0;
                height: auto;
                width: 2000px;
                filter: blur(4px);
                opacity: 0.5;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="rectangle1"></div>', unsafe_allow_html=True)
    st.markdown('<div class="rectangle2"></div>', unsafe_allow_html=True)
    st.markdown('<img class="backgroundLDA" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUk32V2VSgcv0KmlkNtbwmmio1A1R3royvBAq8mjdwbSrXm1FM8Uu3NqWAC-WD6LgAC6M&usqp=CAU" alt="" srcset="">', unsafe_allow_html=True)

    # Đọc HTML từ tập tin và hiển thị
    with codecs.open("index.html", "r", "utf-8") as f:
        html_content = f.read()
    st.markdown(html_content, unsafe_allow_html=True)

def createClock(current_PO4T1, current_MKN, current_IDtocdoquat):

    # Biểu đồ dạng gauge
    data = [
        {
            'type': 'indicator',
            'mode': 'gauge+number+delta',
            'value': 420,
            'title': {'text': 'Giá trị hiện tại', 'font': {'size': 24}},
            'gauge': {
                'axis': {'range': [None, 1000], 'tickwidth': 1, 'tickcolor': 'darkblue'},
                'bar': {'color': 'darkblue'},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': 'gray',
                'steps': [
                    {'range': [0, 250], 'color': 'cyan'},
                    {'range': [250, 400], 'color': 'royalblue'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 490
                }
            }
        }
    ]

    # Layout
    layout = go.Layout(
        width=300,
        height=200,
        margin=dict(t=70, r=25, l=25, b=25),
        paper_bgcolor='white',
        font=dict(color='darkblue', family='Arial')
    )

    # Tạo figure
    fig_clock1 = go.Figure(data=data, layout=layout)
    fig_clock1.data[0]['value'] = current_PO4T1
    fig_clock1.data[0]['title'] = 'Giá trị hiện tại PO4T1'
    fig_clock1.data[0]['gauge']['axis']['range'] = [None, 1500]

    fig_clock2 = go.Figure(data=data, layout=layout)
    fig_clock2.data[0]['value'] = current_MKN
    fig_clock2.data[0]['title'] = 'Giá trị hiện tại MKN'
    fig_clock2.data[0]['gauge']['axis']['range'] = [None, 1]

    fig_clock3 = go.Figure(data=data, layout=layout)
    fig_clock3.data[0]['value'] = current_IDtocdoquat
    fig_clock3.data[0]['title'] = 'Giá trị hiện tại ID tốc độ quạt'
    fig_clock3.data[0]['gauge']['axis']['range'] = [None, 120]

    col1, col2, col3, col4 = st.columns([0.5, 1, 1, 1])
    with col2:
        st.plotly_chart(fig_clock1)
    with col3:
        st.plotly_chart(fig_clock2)
    with col4:
        st.plotly_chart(fig_clock3)

def getDataPredict():
    # Gọi hàm main từ module trainModelProphet
    result = main()
    
    predict_values_mkn = pd.to_datetime(result['mkn']['ds']).dt.strftime('%m/%d/%Y %H:%M')
    predict_values_mkn_yhat = result['mkn']['yhat']

    predict_values_PO4T1 = pd.to_datetime(result['PO4T1']['ds']).dt.strftime('%m/%d/%Y %H:%M')
    predict_values_PO4T1_yhat = result['PO4T1']['yhat']

    predict_values_idtocdoquat = pd.to_datetime(result['idtocdoquat']['ds']).dt.strftime('%m/%d/%Y %H:%M')
    predict_values_idtocdoquat_yhat = result['idtocdoquat']['yhat']

    return result,predict_values_mkn, predict_values_mkn_yhat,predict_values_PO4T1,predict_values_PO4T1_yhat,predict_values_idtocdoquat,predict_values_idtocdoquat_yhat

def showChart(result,predict_values_mkn, predict_values_mkn_yhat,predict_values_PO4T1,predict_values_PO4T1_yhat,predict_values_idtocdoquat,predict_values_idtocdoquat_yhat):

    #chart 1
    fig1 = px.line(df, x='datetime', y='PO4T1', title='Biểu đồ PO4T1')
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    predict_values_PO4T1 = pd.to_datetime(predict_values_PO4T1)
    fig1.update_xaxes(
        title_text='DateTime',
        tickangle=0,  
        nticks=6,  
        tickformat="%Y-%m-%d",
    )
    fig1.update_yaxes(rangemode="tozero")
    fig1.update_traces(line_color='red')
    fig1.add_trace(go.Scatter(x=df1['datetime'], y=df1['PO4T1'], mode='lines', name='Dữ liệu thật', line_color='black'))
    fig1.add_trace(go.Scatter(x=predict_values_PO4T1, y=predict_values_PO4T1_yhat, mode='lines', name='Dữ liệu tiên đoán', line_color='green'))
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    predict_values_PO4T1[0] = pd.to_datetime(predict_values_PO4T1[0])



    #chart 2
    fig2 = px.line(df, x='datetime', y='Oxi', title='Biểu đồ Oxi')
    fig2.update_xaxes(
        title_text='DateTime',
        tickangle=0,
        nticks=7,
        tickformat="%Y-%m-%d",
    )
    fig2.update_yaxes(rangemode="tozero")
    fig2.update_traces(line_color='yellow')
    fig2.add_trace(go.Scatter(x=df1['datetime'], y=df1['Oxi'], mode='lines', name='Dữ liệu thật', line_color='red'))

    #chart 3
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    predict_values_mkn = pd.to_datetime(predict_values_mkn)
    fig3 = px.line(df, x='datetime', y='MKN', title='Biểu đồ MKN')
    fig3.update_xaxes(
        title_text='DateTime',  
        tickangle=0,  
        nticks=7, 
        tickformat="%Y-%m-%d",
    )
    fig3.update_yaxes(rangemode="tozero")
    fig3.update_traces(line_color='black')  
    fig3.add_trace(go.Scatter(x=df1['datetime'], y=df1['MKN'], mode='lines', name='Dữ liệu thật', line_color='blue'))
    fig3.add_trace(go.Scatter(x=predict_values_mkn, y=predict_values_mkn_yhat, mode='lines', name='Dữ liệu tiên đoán', line_color='green'))

    #chart 4
    df1['datetime'] = pd.to_datetime(df1['datetime'])
    predict_values_idtocdoquat = pd.to_datetime(predict_values_idtocdoquat)
    fig4 = px.line(df, x='datetime', y='ID toc do quat', title='Biểu đồ ID tốc độ quạt')
    fig4.update_xaxes(
        title_text='DateTime', 
        tickangle=0,  
        nticks=7,  
        tickformat="%Y-%m-%d",
    )
    fig4.update_yaxes(rangemode="tozero")
    fig4.update_traces(line_color='purple')  
    fig4.add_trace(go.Scatter(x=df1['datetime'], y=df1['ID toc do quat'], mode='lines', name='Dữ liệu thật', line_color='blue'))
    fig4.add_trace(go.Scatter(x=predict_values_idtocdoquat, y=predict_values_idtocdoquat_yhat, mode='lines', name='Dữ liệu tiên đoán', line_color='green'))


    # Hiển thị biểu đồ và thông tin
    col1, col2, col3 = st.columns([0.2, 1, 1])
    with col2:
        st.plotly_chart(fig1)
        st.header(f"Accuracy PO4T1: {result['accuracy_PO4T1']:.2f}%")
        st.plotly_chart(fig2)
        st.header('Accuracy Oxi:')
    with col3:
        st.plotly_chart(fig3)
        st.header(f"Accuracy MKN: {result['accuracy_mkn']:.2f}%")
        st.plotly_chart(fig4)
        st.header(f"Accuracy ID tốc độ quạt: {result['accuracy_idtocdoquat']:.2f}%")


def classify_error(X):
    Fault = []
    Status_Fault = []
    if X[0] < 0:
        Fault.append("Sự cố ngừng mỏ đốt V19")
        Status_Fault.append("Dừng quạt khí")
    else:
        Fault.append("Sự cố ngừng mỏ đốt V19")
        Status_Fault.append("Mỏ đốt V19 hoạt động bình thường")
    if 0 <= X[1] <= 0.1:
        Fault.append("Sự cố ngừng quạt ID fan")
        Status_Fault.append("Lỗi biến tần")
    elif X[1] < 0:
        Fault.append("Sự cố ngừng quạt ID fan")
        Status_Fault.append("Số liệu nhập ID không phù hợp")
    else:
        Fault.append("Sự cố ngừng quạt ID fan")
        Status_Fault.append("Quạt ID fan hoạt động bình thường")
    if X[2] < 45 and X[3] >= 45:
        Fault.append("Sự cố dừng hệ thống vận chuyển thu hồi bụi")
        Status_Fault.append("Cháy động cơ quạt S020A")
    elif X[2] >= 45 and X[3] < 45:
        Fault.append("Sự cố dừng hệ thống vận chuyển thu hồi bụi")
        Status_Fault.append("Cháy động cơ quạt S020B")
    elif X[2] < 45 and X[3] < 45:
        Fault.append("Sự cố dừng hệ thống vận chuyển thu hồi bụi")
        Status_Fault.append("Cháy động cơ quạt S020A, S020B")
    else:
        Fault.append("Sự cố dừng hệ thống vận chuyển thu hồi bụi")
        Status_Fault.append("Hệ thống vận chuyển thu hồi bụi hoạt động bình thường")
    if 0 <= X[4] <= 0.2 and X[5] > 0.2:
        Fault.append("Sự cố vận chuyển liệu")
        Status_Fault.append("Gầu nâng M1 bị nhảy dừng")
    elif X[4] > 0.2 and 0 <= X[5] <= 0.2:
        Fault.append("Sự cố vận chuyển liệu")
        Status_Fault.append("Gầu nâng M2 bị nhảy dừng")
    elif 0 <= X[4] <= 0.2 and 0 <= X[5] <= 0.2:
        Fault.append("Sự cố vận chuyển liệu")
        Status_Fault.append("Gầu nâng M1, M2 bị nhảy dừng")
    elif X[4] < 0 or X[5] < 0:
        Fault.append("Sự cố vận chuyển liệu")
        Status_Fault.append("Số liệu nhập vào không phù hợp với hệ thống")
    else:
        Fault.append("Sự cố vận chuyển liệu")
        Status_Fault.append("Hệ thống vận chuyển liệu hoạt động bình thường")
    if 150 <= X[6] <= 300:
        Fault.append("Mất nước cấp tuần hoàn từ khu D09")
        Status_Fault.append("Một trong hai bơm thuộc hệ thống bơm bị nhảy dừng")
    elif X[6] <= 150:
        Fault.append("Mất nước cấp tuần hoàn từ khu D09")
        Status_Fault.append("Toàn bộ hệ thống bơm bị nhảy dừng")
    else:
        Fault.append("Sự cố vận chuyển liệu")
        Status_Fault.append("Hệ thống cấp nước tuần hoàn từ khu D09 hoạt động bình thường")
    if X[7] > 1100:
        Fault.append("Hư hệ thống cấp liệu")
        Status_Fault.append("""+ Tắt liệu bồn chứa LO1 xuống băng tải S003, S004
                               + Vít S005 chạy lâu ngày bị đóng bám dẫn đến bị dừng
                               + Đứt bằng tải S003,S004
                               + Hư động cơ, hộp giảm tốc S003,S004,S005
                               + Gãy trục rulo băng tải""")
    elif X[7] < 850:
        Fault.append("Hư hệ thống cấp liệu")
        Status_Fault.append("Hệ thống cấp liệu hoạt động bất thường")
    else:
        Fault.append("Hư hệ thống cấp liệu")
        Status_Fault.append("Hệ thống cấp liệu hoạt động bình thường")
    return Fault, Status_Fault

def createTable(Fault, Status_Fault):
    # Tạo DataFrame
    df3 = pd.DataFrame({
        'Loại lỗi': Fault,
        'Trạng thái lỗi hoặc nguyên nhân lỗi': Status_Fault
    })
    return df3

def task2():
    st.write("Parameter")
    with st.form("Form1"):
        # Divide 3 columns
        col1, col2, col3 = st.columns(3)
        # Column 1
        with col1:
            PT0013 = st.text_input("PT0013")
            M1 = st.text_input("ID")
            S020B = st.text_input("S020A")
        # Column 2
        with col2:
            ID = st.text_input("S020B")
            S020A = st.text_input("M1")
            D09 = st.text_input("M2")
        # Column 3
        with col3:
            M2 = st.text_input("D09")
            PO4T1 = st.text_input("PO4T1")
        result = st.form_submit_button("Result")
        dict_items = [
            ("PT0013", PT0013),
            ("ID", ID),
            ("S020A", S020A),
            ("S020B", S020B),
            ("M1", M1),
            ("M2", M2),
            ("D09", D09),
            ("PO4T1", PO4T1)]
    if result:
        all_values_of_dict = True
        for key, value in dict_items:
            if not value:
                st.error(f"Vui lòng nhập số liệu vào ô {key}")
                all_values_of_dict = False
        if all_values_of_dict:
            X_str = [PT0013, ID, S020A, S020B, M1, M2, D09, PO4T1]
            X_int = [int(value) for value in X_str]
            X_arr = np.array(X_int)
            Fault, Status_Fault = classify_error(X_arr)
            df = createTable(Fault, Status_Fault)
            st.table(df)


# task 1
def calculateAccuracy(y_true, y_pred):
    accuracyMultioutput = []
    for i in range(0, len(y_true)):
        delta = abs(y_pred[i] - y_true[i])
        accuracyMultioutput.append(100 - (delta * 100) / abs(y_pred[i]))
    return accuracyMultioutput

def task1():
        # Load model
        filename = '/home/linhtt/Documents/ProjectLDA/visualizeData/multioutput_regression_model_LDA'
        loaded_model = pickle.load(open(filename, 'rb'))

        st.write("Parameter")
        with st.form("Form2"):
            # Divide 4 columns
            col11, col12, col13, col14 = st.columns(4)
            with col11:
                um100 = st.text_input("<100um")
                um10 = st.text_input("<10um")
                um120 = st.text_input("<120um")

            with col12:
                um130 = st.text_input("<130um")
                um150 = st.text_input("<150um")
                um15 = st.text_input("<15um")

            with col13:
                um20 = st.text_input("<20um")
                um45 = st.text_input("<45um")
                um50 = st.text_input("<50um")

            with col14:
                um60 = st.text_input("<60um")
                um70 = st.text_input("<70um")
                um80 = st.text_input("<80um")

            # Divide 2 columns
            col21, col22 = st.columns(2)
            with col21:
                # st.write("Thông tin khí than")
                NhietTri_CO = st.text_input("Giá trị nhiệt của khí than CO")
                NhietDo_CO = st.text_input("Nhiệt độ khí CO")
                LuuLuong_CO = st.text_input("Lưu lượng khí than CO")

            with col22:
                # st.write("Hydrat")
                Do_am = st.text_input("Độ ẩm")
                KL_AH = st.text_input("Tổng khối lượng AH sử dụng")
                KT_AO = st.text_input("Tiêu hao khí than trên 1 tấn AO")

            # Divide 4 columns
            col31, col32, col33, col34 = st.columns(4)
            with col31:
                MKN = st.text_input("MKN")
                Oxi = st.text_input("Oxi")
                PO1P2 = st.text_input("PO1P2")

            with col32:
                PO2P2 = st.text_input("PO2P2")
                PO3P2 = st.text_input("PO3P2")
                PO4P = st.text_input("PO4P")

            with col33:
                CO1P2 = st.text_input("CO1P2")
                CO2P2 = st.text_input("CO2P2")
                CO3P2 = st.text_input("CO3P2")

            with col34:
                CO4P2 = st.text_input("CO4P2")
                PO4T1 = st.text_input("PO4T1")
                ID_fan = st.text_input("ID tốc độ quạt")

            result = st.form_submit_button("Result")

            dict_items = {
                ("um100", um100),
                ("um10", um10),
                ("um120", um120),
                ("um130", um130),
                ("um150", um150),
                ("um15", um15),
                ("um20", um20),
                ("um45", um45),
                ("um50", um50),
                ("um60", um60),
                ("um70", um70),
                ("um80", um80),
                ("Giá trị nhiệt của khí than CO", NhietTri_CO),
                ("Nhiệt độ khí CO", NhietDo_CO),
                ("Lưu lượng khí than CO", LuuLuong_CO),
                ("Độ ẩm", Do_am),
                ("Tổng khối lượng AH sử dụng", KL_AH),
                ("Tiêu hao khí than trên 1 tấn AO", KT_AO),

                ("MKN", MKN),
                ("Oxi", Oxi),
                ("PO1P2", PO1P2),
                ("PO2P2", PO2P2),
                ("PO3P2", PO3P2),
                ("PO4P", PO4P),
                ("CO1P2", CO1P2),
                ("CO2P2", CO2P2),
                ("CO3P2", CO3P2),
                ("CO4P2", CO4P2),
                ("PO4T1", PO4T1),
                ("ID tốc độ quạt", ID_fan)}

        if result:
            all_values_of_dict_coHat = True
            for key, value in dict_items:
                if not value:
                    st.error(f"Vui lòng nhập số liệu vào ô :orange[{key}]")
                    all_values_of_dict_coHat = False

            if all_values_of_dict_coHat is True:
                X_str = [NhietTri_CO, NhietDo_CO, LuuLuong_CO, Do_am, um100, um10, um120, um130, um150,
                        um15, um20, um45, um50, um60, um70, um80, KL_AH, KT_AO]
                y_str = [MKN, Oxi, PO1P2, PO2P2, PO3P2, PO4P, CO1P2, CO2P2, CO3P2, CO4P2, PO4T1, ID_fan]
                X_float = [float(value) for value in X_str]
                y_float = [float(value) for value in y_str]

                X_arr_1D = np.array(X_float)
                y_arr_1D = np.array(y_float)

                result = loaded_model.predict([X_arr_1D])
                result_1D = np.reshape(result, -1)

                accuracyMultioutput = calculateAccuracy(y_arr_1D, result_1D)
                matrix = np.concatenate((result, [y_arr_1D], [accuracyMultioutput]), axis=0)
                df = pd.DataFrame(matrix,
                                columns=['MKN', 'Oxi', 'PO1P2', 'PO2P2', 'PO3P2',
                                        'PO4P', 'CO1P2', 'CO2P2', 'CO3P2', 'CO4P2', 'PO4T1',
                                        'ID toc do quat'])
                st.table(df)


if __name__ == "__main__":
    df, df1 = loadCSV()
    current_PO4T1, current_MKN, current_IDtocdoquat = readLastCSV(df)
    main_dashboard()
    createClock(current_PO4T1, current_MKN, current_IDtocdoquat)
    getDataPredict()
    result,predict_values_mkn, predict_values_mkn_yhat,predict_values_PO4T1,predict_values_PO4T1_yhat,predict_values_idtocdoquat,predict_values_idtocdoquat_yhat = getDataPredict()
    showChart(result,predict_values_mkn, predict_values_mkn_yhat,predict_values_PO4T1,predict_values_PO4T1_yhat,predict_values_idtocdoquat,predict_values_idtocdoquat_yhat)
    task2()
    task1()
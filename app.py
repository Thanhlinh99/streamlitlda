
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
    df = pd.read_csv("data_train1.csv")
    # Đọc dữ liệu từ tập tin CSV add thêm để kiểm tra độ chính xác
    df1 = pd.read_csv('data_add_streamlit.csv')
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


if __name__ == "__main__":
    df, df1 = loadCSV()
    current_PO4T1, current_MKN, current_IDtocdoquat = readLastCSV(df)
    main_dashboard()
    createClock(current_PO4T1, current_MKN, current_IDtocdoquat)
    getDataPredict()
    result,predict_values_mkn, predict_values_mkn_yhat,predict_values_PO4T1,predict_values_PO4T1_yhat,predict_values_idtocdoquat,predict_values_idtocdoquat_yhat = getDataPredict()
    showChart(result,predict_values_mkn, predict_values_mkn_yhat,predict_values_PO4T1,predict_values_PO4T1_yhat,predict_values_idtocdoquat,predict_values_idtocdoquat_yhat)
    task2()

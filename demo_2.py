import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas.io.stata import precision_loss_doc
import pandas_datareader.data as web
import yfinance as yf
from scipy.stats import norm
import time
from BSM_Eu import *
import requests
import pandas as pd
from plotly.subplots import make_subplots

url = 'https://www.slickcharts.com/sp500'
headers = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

request = requests.get(url, headers = headers)

data = pd.read_html(request.text)[0]
data.Symbol = data.Symbol.apply(lambda x: x.replace('.', '-'))
label = data['Symbol']+' - '+data['Company']
# 股票代碼
stk_list = data.Symbol

html_header = """
<head>
<title>PControlDB</title>
<meta charset="utf-8">
<meta name="keywords" content="Stock & Option, ucwzvb, management, EVA">
<meta name="description" content="Stock & Option ucwzvb">
<meta name="author" content="ucwzvb">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<h1 style="font-size:300%; color:#008080; font-family:Georgia"> goohoo <br>
 <h2 style="font-size:50%;color:#008080; font-family:Georgia"> First and Only Web App for Stock option pricing</h3> <br>
 <hr style= "  display: block;
  margin-top: 1em;
  margin-bottom: 1em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 0.25px;"></h1>
"""
st.set_page_config(page_title="Woohoo Finance", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>', unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)
# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)
# sidebar option
st.sidebar.title("Parameters")
Label = st.sidebar.selectbox("Stock",label)
Stock = Label.split(' - ')[0]
type_C = st.sidebar.selectbox("option type",
                    ("Call option","Put option"))

#
type_o = "C" if type_C == 'Call option' else "P"


today = datetime.now()
one_year_ago = today.replace(year=today.year - 1)

stock = yf.Ticker(ticker = Stock)
df = stock.history(start=one_year_ago, end=today, interval="1d")

df = df.sort_values(by='Date')
df = df.dropna()
df = df.assign(close_day_before=df.Close.shift(1))
df['returns'] = ((df.Close - df.close_day_before) / df.close_day_before)
df['simple_rtn'] = df['returns'].pct_change()

sigma = np.sqrt(252) * df['returns'].std()

r = yf.Ticker(ticker = '^TNX')
uty = (r.history(start=today.replace(day=today.day-4),end=today.replace(day=today.day-3))['Close'].iloc[-1])/100
lcp = df['Close'].iloc[-1]


fin = stock.financials
earn = stock.earnings
options = stock.options
asset = stock.balance_sheet.loc['Total Liab']/stock.balance_sheet.loc['Total Assets']

df1 = pd.DataFrame(options)
df1.rename(columns={0: 'dates'}, inplace=True)
df1['dates_2'] = pd.to_datetime(df1['dates'])
dd = df1['dates_2'].to_list()
Date = st.sidebar.selectbox("Strike date",dd)
### Block 1#########################################################################################
with st.container():
    col1, col2, col3, col4, col5, col6, col7 = st.beta_columns([1,15,1,15,1,15,1])
    with col1:
        st.write("")
    with col2:
        st.markdown("""
            <div class="card">
              <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #eef9ea; padding-top: 5px; width: 350px;
               height: 50px;">
                <h3 class="card-title" style="background-color:#eef9ea; color:#008080; font-family:Georgia; text-align: center; padding: 0px 0;"> {}</h3>
              </div>
            </div>
            """.format(Label.split(' - ')[1]), unsafe_allow_html=True)
        st.metric()

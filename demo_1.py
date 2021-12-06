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
headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

request = requests.get(url, headers=headers)

data = pd.read_html(request.text)[0]
data.Symbol = data.Symbol.apply(lambda x: x.replace('.', '-'))
label = data['Symbol'] + ' - ' + data['Company']
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
<h1 style="font-size:300%; color:#008080; font-family:Georgia"> Woohoo <br>
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
st.markdown("<span style=“background-color:#fbfff0”>", unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
# sidebar option
st.sidebar.title("Parameters")
Label = st.sidebar.selectbox("Stock", label)
Stock = Label.split(' - ')[0]
type_C = st.sidebar.selectbox("option type",
                              ("Call option", "Put option"))



today = datetime.now()
one_year_ago = today.replace(year=today.year - 1)

stock = yf.Ticker(ticker=Stock)
df = stock.history(start=one_year_ago, end=today, interval="1d")

df = df.sort_values(by='Date')
df = df.dropna()
df = df.assign(close_day_before=df.Close.shift(1))
df['returns'] = ((df.Close - df.close_day_before) / df.close_day_before)
df['simple_rtn'] = df['returns'].pct_change()

sigma = np.sqrt(252) * df['returns'].std()

r = yf.Ticker(ticker='^TNX')
uty = (r.history(start=today.replace(day=today.day-4),end=today.replace(day=today.day-3))['Close'].iloc[-1]) / 100
lcp = df['Close'].iloc[-1]

fin = stock.financials
earn = stock.earnings
options = stock.options
asset = stock.balance_sheet.loc['Total Liab'] / stock.balance_sheet.loc['Total Assets']

df1 = pd.DataFrame(options)
df1.rename(columns={0: 'dates'}, inplace=True)
df1['dates_2'] = pd.to_datetime(df1['dates'])
dd = df1['dates'].to_list()
Date = st.sidebar.selectbox("Strike date", dd)

from st_card import st_card

# block1
with st.container():
    col1, col2 = st.columns([20, 75])
    with col1:
        st_card(Label.split(' - ')[1], value=lcp.round(2), delta=df['returns'].iloc[-1].round(2),
                delta_description='since last day')
        st_card('Debt Asset ratio', value=asset[0].round(4) * 100, unit='%', show_progress=True)
        st_card('Total Cash', value=stock.info['totalCash'] / 1000000, unit='($)', show_progress=True)
    with col2:
        progress_bar = st.sidebar.progress(0)

        st.markdown("<center><font=size=300% color=#008080 > The Stock Price in Trade</font> </center> ", unsafe_allow_html=True)
        df_1 = df[['Open', 'High', 'Low', 'Close']]
        last_rows = pd.DataFrame(df_1.iloc[0]).transpose()
        chart = st.line_chart(last_rows, width=0, height=355)
        for i in range(len(df) - 1):
            new_rows = df_1.iloc[i + 1]
            s = i / (len(df) - 1) * 100
            chart.add_rows(pd.DataFrame(new_rows).transpose())
            last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()

html_br = """
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([50, 50])
    with col1:
        html_subtitle = """
        <h2 style="color:#008080; font-family:Georgia;"> Key Data : </h2>
        """
        st.markdown(html_subtitle, unsafe_allow_html=True)
        st.markdown(""" 
        <table>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th >Previous Close</th>
            <th >{}</th>
            <th >Market Cap</th>
            <th >{}T</th>
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>Open</th>
            <th>{}</th>
            <th>Beta (5Y Monthly)</th>
            <th>{}</th>
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>Bid</th>
            <th>{} x {}</th>
            <th>PE Ratio (TTM)</th>
            <th>{}</th>
            
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>Ask</th>
            <th>{} x {}</th>
            <th>EPS (TTM)</th>
            <th>{}</th>
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>Day's Range</th>
            <th>{} - {}</th>
            <th>Total Revenue</th>
            <th>{}</th>
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>52 Week Change</th>
            <th>{}</th>
            <th>Forward Dividend & Yield</th>
            <th>{}</th>
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>Volume</th>
            <th>{}</th>
            <th>Ex-Dividend Date</th>
            <th>{}</th>
          </tr>
          <tr style="background-color:#eef9ea; color:#008080; font-family:Georgia; font-size: 15px">
            <th>Avg. Volume</th>
            <th>{}</th>
            <th>1y Target Est</th>
            <th>{}</th>
          </tr>
        </table>
        """.format(stock.info['previousClose'],stock.info['marketCap'],stock.info['open'],stock.info['beta'],stock.info['bid'],stock.info['bidSize'],
        stock.info['pegRatio'],stock.info['ask'],stock.info['askSize'],stock.info['forwardEps'],stock.info['dayLow'],stock.info['dayHigh'],
        stock.info['totalRevenue'],stock.info['52WeekChange'],stock.info['fiveYearAvgDividendYield'],stock.info['volume'],stock.info['dividendRate'],
        stock.info['volume24Hr'],stock.info['volume24Hr'],stock.info['targetMedianPrice']),unsafe_allow_html=True)

    with col2:
        trace1 = go.Bar(
            x=earn.index,
            y=earn.Revenue,
            name="Revenue",
            marker=dict(color='#17A2B8', line=dict(color='#FFFFFF', width=0)),
            text="Revenue")
        # 构造 trace2
        trace2 = go.Bar(
            x=earn.index,
            y=earn.Earnings,
            name="Earnings",
            marker=dict(color='#FF4136', line=dict(color='#FFFFFF', width=0)),
            text="Earnings")
        data = [trace1, trace2]
        import plotly.io as pio
        pio.templates.default = "none"
        layout = go.Layout(barmode="group", template="none", width=800, height=450, title='Company Finicials', font={'color': "#008080", 'family': "Georgia"},plot_bgcolor="#FFFFFF",)
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
html_br = """
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)

with st.container():
    major_holders = stock.major_holders
    inst = major_holders[major_holders[1].isin(['% of Shares Held by Institutions'])][0].str.strip("%").astype(
        float) / 100
    institutional_holders = stock.institutional_holders
    institutional_holders = institutional_holders.append([{'% Out': inst[1] - institutional_holders['% Out'].sum()}],
                                                         ignore_index=True)
    institutional_holders = institutional_holders.append([{'% Out': 1 - inst[1]}], ignore_index=True)
    institutional_holders.iloc[-2, 0] = 'Other institutional_holders'
    institutional_holders.iloc[-1, 0] = 'Other holders'
    institutional_holders.iloc[-2, 2] = institutional_holders.iloc[-3, 2]
    institutional_holders.iloc[-1, 2] = institutional_holders.iloc[-3, 2]
    institutional_holders.iloc[-2, 1] = institutional_holders.iloc[-2, 3] / institutional_holders.iloc[-3, 3] * \
                                        institutional_holders.iloc[-3, 1]
    institutional_holders.iloc[-1, 1] = institutional_holders.iloc[-1, 3] / institutional_holders.iloc[-3, 3] * \
                                        institutional_holders.iloc[-3, 1]
    institutional_holders.iloc[-2, 4] = institutional_holders.iloc[-2, 3] / institutional_holders.iloc[-3, 3] * \
                                        institutional_holders.iloc[-3, 4]
    institutional_holders.iloc[-1, 4] = institutional_holders.iloc[-1, 3] / institutional_holders.iloc[-3, 3] * \
                                        institutional_holders.iloc[-3, 4]
    col1, col2 = st.columns([50, 50])
    with col1:
        pie1 = institutional_holders.Shares
        pie1_list = institutional_holders.Shares.to_list()  # str(2,4) => str(2.4) = > float(2.4) = 2.4
        labels = institutional_holders.Holder
        # figure
        fig = {
            "data": [
                {
                    "values": pie1_list,
                    "labels": labels,
                    "domain": {"x": [0, .5]},
                    "name": "Number Of holders Rates",
                    "hoverinfo": "label+percent+name",
                    "hole": .3,
                    "type": "pie"
                }, ],
            "layout": {
                "title": "holders  rates",
                "font": { 'color': "#008080", 'family': "Georgia"},
                'template':"none"
            }
        }
        st.plotly_chart(fig)
    news = stock.news
    with col2:
        title = news[0]['title']
        net = news[0]['link']
        link = f'- [{title}]({net})'
        html_subtitle = """
                <h2 style="color:#008080; font-family:Georgia;"> News: </h2>
                """
        st.markdown(html_subtitle, unsafe_allow_html=True)
        st.markdown(link, unsafe_allow_html=True)

        title = news[1]['title']
        net = news[1]['link']
        link = f'- [{title}]({net})'
        st.markdown(link, unsafe_allow_html=True)

        title = news[2]['title']
        net = news[2]['link']
        link = f'- [{title}]({net})'
        st.markdown(link, unsafe_allow_html=True)

        title = news[3]['title']
        net = news[3]['link']
        link = f'- [{title}]({net})'
        st.markdown(link, unsafe_allow_html=True)

        title = news[4]['title']
        net = news[4]['link']
        link = f'- [{title}]({net})'
        st.markdown(link, unsafe_allow_html=True)

        title = news[5]['title']
        net = news[5]['link']
        link = f'- [{title}]({net})'
        st.markdown(link, unsafe_allow_html=True)

        title = news[6]['title']
        net = news[6]['link']
        link = f'- [{title}]({net})'
        st.markdown(link, unsafe_allow_html=True)
html_br = """
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)
opt_E_A = st.selectbox("Select option type", ('American', 'European'))
from BAW_Am import *
from BCS_Eu import *
from BSM_Eu import *
from CRR_Am import *
from CRR_Eu import *
from FD_Eu_Am import *
from LSM_Am import *
from FFT_Eu import *
from FD import *
from MCS import*
from int import*
opt = stock.option_chain(Date)
c = opt.calls
p = opt.puts
df_c = pd.DataFrame(c)
df_p = pd.DataFrame(p)
mat = df1.loc[df1['dates'] == Date, 'dates_2'].iloc[0]
T = abs((mat - today).days) / 360
if opt_E_A == 'European':
    option_df_c = pd.DataFrame(columns=
                               ['Contract',
                                'Expiration',
                                'Price',
                                'Strike',
                                'Converged Call %',
                                'Premium',
                                'Black-Scholes',
                                'Cox-Ross-Rubinstein',
                                'Monte Carlo Simulation',
                                'finite difference method',
                                'Fourier-based approach',
                                'Difference',
                                'Volume',
                                'Open Interest',
                                'Impelied Volatility', ])
    option_df_p = pd.DataFrame(columns=['Contract',
                                        'Expiration',
                                        'Price',
                                        'Strike',
                                        'Converged Call %',
                                        'Premium',
                                        'Black-Scholes',
                                        'Cox-Ross-Rubinstein',
                                        'Monte Carlo Simulation',
                                        'finite difference method',
                                        'Fourier-based approach',
                                        'Difference',
                                        'Volume',
                                        'Open Interest',
                                        'Impelied Volatility', ])

    for K in df_c['strike']:
        market_premium = df_c.loc[df_c['strike'] == K, 'lastPrice'].iloc[0]
        c_call = ((K / (lcp - market_premium)) - 1)
        sigma_iv = df_c.loc[df_c['strike'] == K, 'impliedVolatility'].iloc[0]
        volume = df_c.loc[df_c['strike'] == K, 'volume'].iloc[0]
        inTheMoney = df_c.loc[df_c['strike'] == K, 'inTheMoney'].iloc[0]
        premium_call_iv = BSM_Eu(lcp, K, uty, T, sigma, 'C')
        premium_call_crr = CRR_option_value(lcp, K, uty, T, sigma, 'C', round(T * 360) * 2)
        [v0, ci] = monte_carlo_bs_eu(lcp, K, uty, 0, sigma, T, 5000000, 'call', True)
        premium_call_mcs = v0
        premium_call_FFT = fast_fourier_bs_eu(lcp, K, uty, sigma, T, option_type='C', n=10000, m=400, t=0)
        difference_iv = market_premium - premium_call_iv
        option_df_c = option_df_c.append(
            {
                'Contract': df_c.loc[df_c['strike'] == K, 'contractSymbol'].iloc[0],
                'Expiration': Date,
                'Price': lcp,
                'Strike': K,
                'Converged Call %': c_call,
                'Premium': market_premium,
                'Black-Scholes': premium_call_iv,
                'Cox-Ross-Rubinstein': premium_call_crr,
                'Monte Carlo Simulation': premium_call_mcs,
                'Fourier-based approach': premium_call_FFT,
                'Difference': difference_iv,
                'Volume': df_c.loc[df_c['strike'] == K, 'volume'].iloc[0],
                'Open Interest': df_c.loc[df_c['strike'] == K, 'openInterest'].iloc[0],
                'Impelied Volatility': df_c.loc[df_c['strike'] == K, 'impliedVolatility'].iloc[0],
            },
            ignore_index=True
        )
    for K in df_p['strike']:
        market_premium = df_p.loc[df_p['strike'] == K, 'lastPrice'].iloc[0]

        p_call = ((K / (lcp - market_premium)) - 1)
        sigma_iv = df_p.loc[df_p['strike'] == K, 'impliedVolatility'].iloc[0]
        volume = df_p.loc[df_p['strike'] == K, 'volume'].iloc[0]
        inTheMoney = df_p.loc[df_p['strike'] == K, 'inTheMoney'].iloc[0]
        premium_put_iv = BSM_Eu(lcp, K, uty, T, sigma, 'P')
        premium_put_crr = CRR_option_value(lcp, K, uty, T, sigma, 'P', round(T * 360))
        [v0, ci] = monte_carlo_bs_eu(lcp, K, uty, 0, sigma, T, 5000000, 'put', True)
        premium_put_mcs = v0
        premium_put_FFT = fast_fourier_bs_eu(lcp, K, uty, sigma, T, option_type='C', n=10000, m=400, t=0)
        difference_iv = market_premium - premium_put_iv
        option_df_p = option_df_p.append(
            {
                'Contract': df_p.loc[df_p['strike'] == K, 'contractSymbol'].iloc[0],
                'Expiration': Date,
                'Price': lcp,
                'Strike': K,
                'Converged Call %': p_call,
                'Premium': market_premium,
                'Black-Scholes': premium_put_iv,
                'Cox-Ross-Rubinstein': premium_put_crr,
                'Monte Carlo Simulation': premium_put_mcs,
                'Fourier-based approach': premium_put_FFT,
                'Difference': difference_iv,
                'Volume': df_p.loc[df_p['strike'] == K, 'volume'].iloc[0],
                'Open Interest': df_p.loc[df_p['strike'] == K, 'openInterest'].iloc[0],
                'Impelied Volatility': df_p.loc[df_p['strike'] == K, 'impliedVolatility'].iloc[0],
            },
            ignore_index=True
        )
    if type_C == 'Call option':
        st.markdown(f"The Call Option Price in Trade {Date}", unsafe_allow_html=True)
        option_df_c.index = option_df_c['Strike']
        df_1 = option_df_c[['Premium', 'Black-Scholes', 'Cox-Ross-Rubinstein', 'Monte Carlo Simulation','Fourier-based approach']]
        last_rows = pd.DataFrame(df_1.iloc[0]).transpose()
        chart = st.line_chart(last_rows, width=0, height=355)
        for i in range(len(option_df_c) - 1):
            new_rows = df_1.iloc[i + 1]
            s = i / (len(option_df_c) - 1) * 100
            chart.add_rows(pd.DataFrame(new_rows).transpose())
            last_rows = new_rows
            time.sleep(0.05)
    else:
        st.markdown(f"The put Option Price in Trade {Date}", unsafe_allow_html=True)
        option_df_p.index = option_df_p['Strike']
        df_1 = option_df_p[
            ['Premium', 'Black-Scholes', 'Cox-Ross-Rubinstein', 'Monte Carlo Simulation',
             'Fourier-based approach']]
        last_rows = pd.DataFrame(df_1.iloc[0]).transpose()
        chart = st.line_chart(last_rows, width=0, height=355)
        for i in range(len(option_df_p) - 1):
            new_rows = df_1.iloc[i + 1]
            s = i / (len(option_df_p) - 1) * 100
            chart.add_rows(pd.DataFrame(new_rows).transpose())
            last_rows = new_rows
            time.sleep(0.05)
else:
    option_df_p_Eu = pd.DataFrame(columns=['Contract',
                                           'Expiration',
                                           'Price',
                                           'Strike',
                                           'Converged Call %',
                                           'Premium',
                                           'Black-Scholes',
                                           'Cox-Ross-Rubinstein',
                                           'Monte Carlo Simulation',
                                           'finite difference method',
                                           'Fourier-based approach',
                                           'Difference',
                                           'Volume',
                                           'Open Interest',
                                           'Impelied Volatility', ])

    option_df_p = pd.DataFrame(columns=['Contract',
                                        'Expiration',
                                        'Price',
                                        'Strike',
                                        'Converged Call %',
                                        'Premium',
                                        'Barone-Adesi',
                                        'Cox-Ross-Rubinstein',
                                        'Monte Carlo Simulation',
                                        'finite difference method',
                                        'Difference',
                                        'Volume',
                                        'Open Interest',
                                        'Impelied Volatility', ])

    for K in df_p['strike']:
        market_premium = df_p.loc[df_p['strike'] == K, 'lastPrice'].iloc[0]

        p_call = ((K / (lcp - market_premium)) - 1)
        sigma_iv = df_p.loc[df_p['strike'] == K, 'impliedVolatility'].iloc[0]
        volume = df_p.loc[df_p['strike'] == K, 'volume'].iloc[0]
        inTheMoney = df_p.loc[df_p['strike'] == K, 'inTheMoney'].iloc[0]
        premium_put_crr = CRR_option_valuation_Am(lcp, K, uty, T, sigma, 100)
        paths = sim_gbm_paths(lcp, sigma, T, uty, round(T*360), 5000000, 0, True)
        [v0, se] = monte_carlo_bs_am(K, uty, T, 'put', paths, 4, "laguerre", 'svd')
        premium_put_mcs = v0
        premium_put_BAW = getValue('American', 'Value', 'Put', lcp, K, T, uty, uty, sigma)
        option_df_p = option_df_p.append(
            {
                'Contract': df_p.loc[df_p['strike'] == K, 'contractSymbol'].iloc[0],
                'Expiration': Date,
                'Price': lcp,
                'Strike': K,
                'Converged Call %': p_call,
                'Premium': market_premium,
                'Barone-Adesi': premium_put_BAW,
                'Cox-Ross-Rubinstein': premium_put_crr,
                'Monte Carlo Simulation': premium_put_mcs,
                'Volume': df_p.loc[df_p['strike'] == K, 'volume'].iloc[0],
                'Open Interest': df_p.loc[df_p['strike'] == K, 'openInterest'].iloc[0],
                'Impelied Volatility': df_p.loc[df_p['strike'] == K, 'impliedVolatility'].iloc[0],
            },
            ignore_index=True
        )
    st.markdown(f"The put Option Price in Trade {Date}", unsafe_allow_html=True)
    option_df_p.index = option_df_p['Strike']
    df_1 = option_df_p[
        ['Premium', 'Barone-Adesi', 'Cox-Ross-Rubinstein', 'Monte Carlo Simulation']]
    last_rows = pd.DataFrame(df_1.iloc[0]).transpose()
    chart = st.line_chart(last_rows, width=0, height=355)
    for i in range(len(option_df_p) - 1):
        new_rows = df_1.iloc[i + 1]
        s = i / (len(option_df_p) - 1) * 100
        chart.add_rows(pd.DataFrame(new_rows).transpose())
        last_rows = new_rows
        time.sleep(0.05)

html_line="""
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;">
"""
st.markdown(html_line, unsafe_allow_html=True)
null8_0,row8_1= st.columns((0.09,4))

with row8_1:
    st.write(
    """
    ### **Contacts**
    [![](https://img.shields.io/badge/GitHub-Follow-informational)](https://github.com/ucwzvb/Woohoo-finance)
    [![](https://img.shields.io/badge/Open-Issue-informational)](https://github.com/ucwzvb/Woohoo-finance/issues)
    [![MAIL Badge](https://img.shields.io/badge/-crisjoe621@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:crisjoe621@gmail.com)](mailto:crisjoe621@gmail.com)
    ##### © ucwzvb, 2021
    """
)
#-- Modules & Packages
import pandas as pd 
import numpy as np 

from glob import glob 
from time import strftime, sleep 
from datetime import datetime
from pandas_datareader import data as pdr
from pandas.tseries.offsets import BDay 

import yfinance as yf
yf.pdr_override()

#-- Plotting 

import dash 
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html 
import dash_bootstrap_components as dbc 
import plotly.express as px 
import dash_table 
import plotly.graph_objects as go 
from jupyter_dash import JupyterDash



portfolio_tickers = ['BTC-USD', 'ETH-USD', 'OP-USD','ARB-USD','AVAX-USD']

one_year = datetime.today().year - 1
today = datetime.today()
start_x = datetime(2022, 1, 1)
end_x = today

start_assets = datetime(2022,1,1)
end_assets = today

start_ytd = datetime(one_year, 12, 31) + BDay(1)

# def get_txs_data(tickers, startDate, interval):
def data_x(ticker):
    #-- monthly interval '1mo'
    data = (yf.download(ticker, start = start_x, interval= '1mo'))
    data.columns = [x.lower() for x in data.columns]
    data.drop(columns = ['open','high','low','volume','adj close'], axis = 1, inplace = True)
    data['ticker'] =  ticker
    data['type'] = 'Buy'
    data['val_transact'] = 50000
    data['quantity'] = data['val_transact']/ data['close'] 
    data['prev_units'] = data['quantity'].shift(1)
    data['cml_units'] = data['quantity'].cumsum()
    data['prev_cost'] = data['val_transact'].shift(1)
    data['cml_cost'] = data['val_transact'].cumsum()
    #-- applying spot taker-fees in Binance Spot Markets
    data['cost_transact'] = data['val_transact'] * 0.0009500
    data['cml_invested'] = data['val_transact'].cumsum() - data['cost_transact']
    data['cost_unit'] = data['cml_cost'] / data['quantity']
    data['cum_position_val'] = data['cml_units'] * data['close']
    data['gain_loss'] = data['cum_position_val'] - data['cml_invested'] 
    data['yield'] = (data['gain_loss'] / data['cml_invested']) - 1
    #-- running, cumulative mean to accurately assess the avg price through each purchase date/row
    data['avg_price'] = data['close'].expanding().mean()
    data['current_value'] = data['close'] * data['cml_units']

    #-- Assigning recurrent exact amount for each crypto asset
    if ticker == 'BTC-USD':
      data['val_transact'] = 50000
    elif ticker == 'ETH-USD':
      data['val_transact'] = 50000
    elif ticker == 'AVAX-USD':
      data['val_transact'] = 10000
    elif ticker == 'LTC-USD':
      data['val_transact'] = 10000
    return data

#-- Fetching data for BTC-USD
btcusd = data_x('BTC-USD')
#-- Fetching data for ETH-USD
ethusd = data_x('ETH-USD')
#-- Fetching data for LTC-USD
ltcusd = data_x('LTC-USD')
#-- Fetching data for AVAX-USD
avaxusd = data_x('AVAX-USD')

first_concat =  pd.concat([btcusd, ethusd])
second_concat = pd.concat([ltcusd, avaxusd])

# all_df = [first_concat,second_concat]
# transactions_df = pd.concat(all_df)
transactions_df = pd.concat([first_concat, second_concat])
#-- Saving transactions dataframe
transactions_df.to_excel(r"dummy_transactions.xlsx")
#-- Previewing transactions dataframe
transactions_df.tail(5) 


#-- Cleaning columns string names
def clean_headers(df):
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '').str.replace('(','').str.replace(')','').str.replace(' ','_').str.replace('_/_','/')

#-- Getting timestamps for file names before saving 
def get_tmstmp():
    now = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    return now

last_file = (r'dummy_transactions.xlsx')
# print(last_file[-(len(last_file)) + (last_file.rfind('/')+1):])

all_ops = pd.read_excel(last_file)
all_ops.date = pd.to_datetime(all_ops.Date, format = "%d/%m/%Y")

all_tickers = list(all_ops['ticker'].unique())
blackList = ['PEPE-USD', 'DOGE-USD', 'ADA-USD']
final_tickers = [tick for tick in all_tickers if tick not in blackList]
print("Traded {} different cryptos".format(len(all_tickers)))
#-- All transactions without blacklisted assets
final_filtered = all_ops[~all_ops.ticker.isin(blackList)]



portfolio_tickers = ['BTC-USD', 'ETH-USD','AVAX-USD', 'LTC-USD']

one_year = datetime.today().year - 1
today = datetime.today()
start_x = datetime(2022, 1, 1)
end_x = today

start_assets = datetime(2022,1,1)
end_assets = today

start_ytd = datetime(one_year, 12, 31) + BDay(1)

def get_data(tickers, startDate, endDate):
    def data(ticker):
        data = (pdr.get_data_yahoo(ticker, start = startDate, end = endDate))
        data.columns = [x.lower() for x in data.columns]
        data.drop(columns = ['adj close'], axis = 1, inplace = True)
        return data
    datum = map(data, tickers)
    return(pd.concat(datum, keys = tickers, names = ['ticker', 'date']))

all_prices_df = get_data(portfolio_tickers, start_x, end_x)
all_prices_df

# #-- Saving all asset prices separately
# for ticker in final_filtered:
#     all_prices_df.loc[ticker].to_csv("D:\_datasets\Projects_Coding\portfolio_dashboard_base2\outputs\{}_price_hist.csv".format(ticker))

# all_prices_df['ticker']

mega_dict = {}  # you have to create it first
min_date = '2022-01-01'  # optional
# TX_COLUMNS = ['date','ticker', 'cashflow', 'cml_units', 'cml_cost', 'gain_loss']
TX_COLUMNS = ['Date','ticker', 'cml_units','val_transact', 'cml_cost', 'gain_loss']

tx_filt = all_ops[TX_COLUMNS]  # keeping just the most relevant ones for now

for ticker in portfolio_tickers:
    prices_df = all_prices_df[all_prices_df.index.get_level_values('ticker').isin([ticker])].reset_index()
    ## Can add more columns like volume!
    PX_COLS = ['date', 'close']
    prices_df = prices_df[PX_COLS].set_index(['date'])
    # Making sure we get sameday transactions
    tx_df = tx_filt[tx_filt.ticker==ticker].groupby('Date').agg({'val_transact': 'sum',
                                                                 'cml_units': 'last',
                                                                 'cml_cost': 'last',
                                                                 'gain_loss': 'sum'})
    # Merging price history and transactions dataframe
    tx_and_prices = pd.merge(prices_df, tx_df, how='outer', left_index=True, right_index=True).fillna(0)
    # This is to fill the days that were not in our transaction dataframe
    tx_and_prices['cml_units'] = tx_and_prices['cml_units'].replace(to_replace=0, method='ffill')
    tx_and_prices['cml_cost'] = tx_and_prices['cml_cost'].replace(to_replace=0, method='ffill')
    tx_and_prices['gain_loss'] = tx_and_prices['gain_loss'].replace(to_replace=0, method='ffill')
    # Cumulative sum for the val_transact
    tx_and_prices['val_transact'] = tx_and_prices['val_transact'].cumsum()
    tx_and_prices['avg_price'] = (tx_and_prices['cml_cost']/tx_and_prices['cml_units'])
    tx_and_prices['mktvalue'] = (tx_and_prices['cml_units']*tx_and_prices['close'])
    tx_and_prices = tx_and_prices.add_prefix(ticker+'_')
    # Once we're happy with the dataframe, add it to the dictionary
    mega_dict[ticker] = tx_and_prices.round(3)
		
# check an individual stock
# MEGA_DICT['RUN'].tail()

# saving it, so we can access it quicker later
mega_dataset = pd.concat(mega_dict.values(), axis=1)
# MEGA_DF.to_csv('../outputs/mega_df/MEGA_DF_{}.csv'.format(get_data(final_tickers, start_x, end_x)))  # optional

# like this:
# last_file = glob('../outputs/mega/MEGA*.csv')[-1] # path to file in the folder
# print(last_file[-(len(last_file))+(last_file.rfind('/')+1):])
# mega_dataset = pd.read_csv(last_file)
# mega_dataset['date'] = pd.to_datetime(mega_dataset['date'])
# mega_dataset.set_index('Date', inplace=True)



portf_allvalues = mega_dataset.filter(regex='mktvalue').fillna(0) #  getting just the market value of each ticker
portf_allvalues['portf_value'] = portf_allvalues.sum(axis=1) # summing all market values

# For the S&P500 price return
sp500 = pdr.get_data_yahoo('^GSPC', start_x, end_x)
# clean_header(sp500)

#getting the pct change
portf_allvalues = portf_allvalues.join(sp500['Close'], how='inner')
portf_allvalues.rename(columns={'Close': 'sp500_mktvalue'}, inplace=True)
portf_allvalues['ptf_value_pctch'] = (portf_allvalues['portf_value'].pct_change()*100).round(2)
portf_allvalues['sp500_pctch'] = (portf_allvalues['sp500_mktvalue'].pct_change()*100).round(2)
portf_allvalues['ptf_value_diff'] = (portf_allvalues['portf_value'].diff()).round(2)
portf_allvalues['sp500_diff'] = (portf_allvalues['sp500_mktvalue'].diff()).round(2)
# KPI's for portfolio
kpi_portfolio7d_abs = portf_allvalues.tail(7).ptf_value_diff.sum().round(2)
kpi_portfolio15d_abs = portf_allvalues.tail(15).ptf_value_diff.sum().round(2)
kpi_portfolio30d_abs = portf_allvalues.tail(30).ptf_value_diff.sum().round(2)
kpi_portfolio200d_abs = portf_allvalues.tail(200).ptf_value_diff.sum().round(2)
kpi_portfolio7d_pct = (kpi_portfolio7d_abs/portf_allvalues.tail(7).portf_value[0]).round(3)*100
kpi_portfolio15d_pct = (kpi_portfolio15d_abs/portf_allvalues.tail(15).portf_value[0]).round(3)*100
kpi_portfolio30d_pct = (kpi_portfolio30d_abs/portf_allvalues.tail(30).portf_value[0]).round(3)*100
kpi_portfolio200d_pct = (kpi_portfolio200d_abs/portf_allvalues.tail(200).portf_value[0]).round(3)*100
# KPI's for S&P500
kpi_sp500_7d_abs = portf_allvalues.tail(7).sp500_diff.sum().round(2)
kpi_sp500_15d_abs = portf_allvalues.tail(15).sp500_diff.sum().round(2)
kpi_sp500_30d_abs = portf_allvalues.tail(30).sp500_diff.sum().round(2)
kpi_sp500_200d_abs = portf_allvalues.tail(200).sp500_diff.sum().round(2)
kpi_sp500_7d_pct = (kpi_sp500_7d_abs/portf_allvalues.tail(7).sp500_mktvalue[0]).round(3)*100
kpi_sp500_15d_pct = (kpi_sp500_15d_abs/portf_allvalues.tail(15).sp500_mktvalue[0]).round(3)*100
kpi_sp500_30d_pct = (kpi_sp500_30d_abs/portf_allvalues.tail(30).sp500_mktvalue[0]).round(3)*100
kpi_sp500_200d_pct = (kpi_sp500_200d_abs/portf_allvalues.tail(200).sp500_mktvalue[0]).round(3)*100



initial_date = '2022-01-01'  # do not use anything earlier than your first trade
plotly_prtfl_val = portf_allvalues[portf_allvalues.index > initial_date]
plotly_prtfl_val = plotly_prtfl_val[['portf_value', 'sp500_mktvalue', 'ptf_value_pctch',
                                     'sp500_pctch', 'ptf_value_diff', 'sp500_diff']].reset_index().round(2)
# calculating cumulative growth since initial date
plotly_prtfl_val['ptf_growth'] = plotly_prtfl_val.portf_value/plotly_prtfl_val['portf_value'].iloc[0]
plotly_prtfl_val['sp500_growth'] = plotly_prtfl_val.sp500_mktvalue/plotly_prtfl_val['sp500_mktvalue'].iloc[0]
plotly_prtfl_val.rename(columns={'index': 'date'}, inplace=True)  # needed for later

# Plotly part
CHART_THEME = 'plotly_white'  # others examples: seaborn, ggplot2, plotly_dark
chart_ptfl_value = go.Figure()  # generating a figure that will be updated in the following lines
chart_ptfl_value.add_trace(go.Scatter(x=plotly_prtfl_val.date, y=plotly_prtfl_val.portf_value,
                    mode='lines',  # you can also use "lines+markers", or just "markers"
                    name='Global Value'))
chart_ptfl_value.layout.template = CHART_THEME
chart_ptfl_value.layout.height=500
chart_ptfl_value.update_layout(margin = dict(t=50, b=50, l=25, r=25))  # this will help you optimize the chart space
chart_ptfl_value.update_layout(
    title='Total BASE 2 Portfolio Value (USD$)',
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='Value: $ USD',
        titlefont_size=14,
        tickfont_size=12,
        ))
chart_ptfl_value.update_xaxes(rangeslider_visible=False)
chart_ptfl_value.update_layout(showlegend=False)
chart_ptfl_value.show()


df = plotly_prtfl_val[['date', 'ptf_growth', 'sp500_growth']].copy().round(3)
df['month'] = df.date.dt.month_name()  # date column should be formatted as datetime
df['weekday'] = df.date.dt.day_name()  # could be interesting to analyze weekday returns later
df['year'] = df.date.dt.year
df['weeknumber'] = df.date.dt.week    # could be interesting to try instead of timeperiod
df['timeperiod'] = df.year.astype(str) + ' - ' + df.date.dt.month.astype(str).str.zfill(2)

# getting the percentage change for each period. the first period will be NaN
sp = df.reset_index().groupby('timeperiod').last()['sp500_growth'].pct_change()*100
ptf = df.reset_index().groupby('timeperiod').last()['ptf_growth'].pct_change()*100
plotlydf_growth_compare = pd.merge(ptf, sp, on='timeperiod').reset_index().round(3)
plotlydf_growth_compare.head()

# Plotly part
fig_growth2 = go.Figure()
fig_growth2.layout.template = CHART_THEME
fig_growth2.add_trace(go.Bar(
    x=plotlydf_growth_compare.timeperiod,
    y=plotlydf_growth_compare.ptf_growth.round(2),
    name='Portfolio'
))
fig_growth2.add_trace(go.Bar(
    x=plotlydf_growth_compare.timeperiod,
    y=plotlydf_growth_compare.sp500_growth.round(2),
    name='S&P 500',
))
fig_growth2.update_layout(barmode='group')
fig_growth2.layout.height=300
fig_growth2.update_layout(margin = dict(t=50, b=50, l=25, r=25))
fig_growth2.update_layout(
    xaxis_tickfont_size=12,
    yaxis=dict(
        title='% change',
        titlefont_size=13,
        tickfont_size=12,
        ))

fig_growth2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99))
fig_growth2.show()



indicators_ptf = go.Figure()
indicators_ptf.layout.template = CHART_THEME

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio7d_pct,
    number = {'suffix': " %"},
    title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_7d_pct, 'relative': False},
    domain = {'row': 0, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio15d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_15d_pct, 'relative': False},
    domain = {'row': 1, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio30d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_30d_pct, 'relative': False},
    domain = {'row': 2, 'column': 0}))

indicators_ptf.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_portfolio200d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
    delta = {'position': "bottom", 'reference': kpi_sp500_200d_pct, 'relative': False},
    domain = {'row': 3, 'column': 1}))

indicators_ptf.update_layout(
    grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
    margin=dict(l=50, r=50, t=30, b=30)
)

indicators_sp500 = go.Figure()
indicators_sp500.layout.template = CHART_THEME
indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_7d_pct,
    number = {'suffix': " %"},
    title = {"text": "<br><span style='font-size:0.7em;color:gray'>7 Days</span>"},
    domain = {'row': 0, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_15d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>15 Days</span>"},
    domain = {'row': 1, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_30d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>30 Days</span>"},
    domain = {'row': 2, 'column': 0}))

indicators_sp500.add_trace(go.Indicator(
    mode = "number+delta",
    value = kpi_sp500_200d_pct,
    number = {'suffix': " %"},
    title = {"text": "<span style='font-size:0.7em;color:gray'>200 Days</span>"},
    domain = {'row': 3, 'column': 1}))

indicators_sp500.update_layout(
    grid = {'rows': 4, 'columns': 1, 'pattern': "independent"},
    margin=dict(l=50, r=50, t=30, b=30)
)


# getting the accumulated positions for our tickers
last_positions = all_ops.groupby(['ticker']).agg({'cml_units': 'last', 'cml_cost': 'last',
                                                'gain_loss': 'sum', 'val_transact': 'sum'}).reset_index()
curr_prices = []

for tick in last_positions['ticker']:
    # stonk = yf.get_data(tick)
    asset = (yf.download(tick, start = today))

    price = asset['Close']
    curr_prices.append(price)
    print(f'Done for {tick}')
		
last_positions['price'] = curr_prices  # adding it to our dataframe
# last_positions['current_value'] = (last_positions.price * last_positions.cml_units).round(2)  # and now we can calculate
last_positions['current_value'] = all_ops['current_value']
# last_positions['avg_price'] = (last_positions.cml_cost / last_positions.cml_units).round(2)  # and now we can calculate
last_positions['avg_price'] = all_ops['avg_price']
last_positions = last_positions.sort_values(by='current_value', ascending=False)  # sorting by current value

# Plotly part
holding_donut_top = go.Figure()
holding_donut_top.layout.template = CHART_THEME
holding_donut_top.add_trace(go.Pie(labels=last_positions.head(15).ticker, values=last_positions.head(15).current_value))
holding_donut_top.update_traces(hole=.4, hoverinfo="label+value+percent")
holding_donut_top.update_traces(textposition='outside', textinfo='label+value')
holding_donut_top.update_layout(showlegend=False)
holding_donut_top.update_layout(margin = dict(t=50, b=50, l=25, r=25))


app = JupyterDash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H2('Base 2 Portfolio Overview', className='text-center text-primary, mb-3'))),  # header row
        
        dbc.Row([  # start of second row
            dbc.Col([  # first column on second row
            html.H5('Total Portfolio Value ($USD)', className='text-center'),
            dcc.Graph(id='chrt-portfolio-main',
                      figure=chart_ptfl_value,
                      style={'height':550}),
            html.Hr(),
            ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on second row
            dbc.Col([  # second column on second row
            html.H5('Portfolio', className='text-center'),
            dcc.Graph(id='indicators-ptf',
                      figure=indicators_ptf,
                      style={'height':550}),
            html.Hr()
            ], width={'size': 2, 'offset': 0, 'order': 2}),  # width second column on second row
            dbc.Col([  # third column on second row
            html.H5('benchmark: S&P500', className='text-center'),
            dcc.Graph(id='indicators-sp',
                      figure=indicators_sp500,
                      style={'height':550}),
            html.Hr()
            ], width={'size': 2, 'offset': 0, 'order': 3}),  # width third column on second row
        ]),  # end of second row
        
        dbc.Row([  # start of third row
            dbc.Col([  # first column on third row
                html.H5('Monthly Return (%)', className='text-center'),
                dcc.Graph(id='chrt-portfolio-secondary',
                      figure=fig_growth2,
                      style={'height':380}),
            ], width={'size': 8, 'offset': 0, 'order': 1}),  # width first column on second row
            dbc.Col([  # second column on third row
                html.H5('Holdings', className='text-center'),
                dcc.Graph(id='pie-top15',
                      figure = holding_donut_top,
                      style={'height':380}),
            ], width={'size': 4, 'offset': 0, 'order': 2}),  # width second column on second row
        ])  # end of third row
        
    ], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True, port=8058)
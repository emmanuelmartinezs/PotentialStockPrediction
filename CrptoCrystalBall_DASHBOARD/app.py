import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from datetime import datetime
from pathlib import Path
from os import listdir
from os.path import isfile, join

# Instantiate dash app
app = dash.Dash()

# Read all .csv files, make a dictionary:
# Key = file name : Value: dataframe
source = Path('datasets')
filenames = [f for f in listdir(source) if isfile(join(source, f))]
datasets = dict()
for filename in filenames:
    filepath = source/filename
    try:  # Date column is 'Date'
        datasets[filepath.name] = (pd.read_csv(filepath, parse_dates = ['Date']))
    except ValueError:  # Date column is 'ds'
        datasets[filepath.name] = (pd.read_csv(filepath, parse_dates = ['ds']))
        datasets[filepath.name].drop(['Unnamed: 0'], axis = 'columns', inplace = True)  # Remove undesired column

# Make options for Dropdown selector
options = list()
for currency in datasets['All_top_coins-updated-21.csv']['Currency Name'].unique():
    options.append({
    'label': currency,
    'value': currency
    })

# Disclaimer on the bottom of Dashboard (Markdown text)
disclaimer = 'Disclaimer: Past performance does not guarantee future results and current performance may be lower or higer than the past performance data quoted.'

# Make a dict to convert currency name into filename
dataset = {
    'Bitcoin': 'forecast_bitcoin.csv',
    'Ethereum': 'forecast_ethereum.csv',
    'Litecoin': 'forecast_litecoin.csv',
    'Ripple': 'forecast_ripple.csv',
    'Tether': 'forecast_tether.csv',
    # If needed, just add new entries
}

# CALLBACKS
@app.callback(
Output('date-picker', 'initial_visible_month'),
[Input('date-picker', 'start_date'),
Input('date-picker', 'end_date')]
)
def update_datepicker_month(start_date, end_date):
    # This makes the calendar open on the end date if start date was picked, and vice-versa
    ctx = dash.callback_context
    if ctx.triggered and (ctx.triggered[0]['prop_id'] == 'date-picker.start_date'):
        return end_date
    elif ctx.triggered and (ctx.triggered[0]['prop_id'] == 'date-picker.end_date'):
        return start_date
    else:
        raise PreventUpdate

@app.callback(
Output('main-graph', 'figure'),
[Input('dropdown', 'value'),
Input('date-picker', 'start_date'),
Input('date-picker', 'end_date')]
)
def make_figure(currency_list, start_date, end_date):
    # Makes the main figure, given currencies and date range
    df = datasets['All_top_coins-updated-21.csv']
    fig = go.Figure()
    fig.update_layout(template = 'plotly_dark', height = 800, plot_bgcolor='black', paper_bgcolor='black')
    colors = dict()  # Color dict to save each trace color by currency
    for index, currency in enumerate(currency_list):
        colors[currency] = px.colors.qualitative.Plotly[index]
        mask = (df['Currency Name'] == currency)
        fig.add_traces(
            go.Scatter(
                x = df.loc[mask,'Date'],
                y = df.loc[mask,'Close'],
                name = currency,
                legendgroup = currency,
                mode = 'lines',
                hovertemplate =
                    'Date: %{x}'+
                    '<br><b>Price</b>: %{y:$,.2f}<br>'+
                    '<b>{} Data</b>'.format(currency)+
                    "<extra></extra>",
            )
        )
    for index, (currency, file) in enumerate(dataset.items()):
        if currency in currency_list:
            forecast_df = datasets[file]
            fig.add_traces(
                go.Scatter(
                    x = forecast_df['ds'],
                    y = forecast_df['yhat'],
                    hovertemplate =
                        'Date: %{x}'+
                        '<br><b>Price</b>: %{y:$,.2f}<br>'+
                        '<b>{} Forecast</b>'.format(currency)+
                        "<extra></extra>",
                    showlegend=False,
                    legendgroup = currency,
                    line_color = colors[currency],
                    line_dash = 'dash',
                    mode = 'lines'
                )
            )
    fig.update_layout(
        xaxis_range = [pd.Timestamp(start_date),pd.Timestamp(end_date)],
        xaxis_title_text = '<b>Date</b>',
        xaxis_title_font_size = 16,
        yaxis_title_text = '<b>Price</b>',
        yaxis_title_font_size = 16,
        yaxis_tickformat = '$,.2'
    )
    return fig

# LAYOUT
app.layout = html.Div(
    [
        html.Div(
            [
                # Add dashboard heading with background overlay
                html.Div(
                    html.H1('Crypto Crystal Ball Dashboard'),
                    className = 'overlay'
                ),
                # Add a wrapper Div for two columns
                html.Div(
                    [
                        # Add first column
                        html.Div(
                            [
                                # Add cryptocurrency dropdown
                                html.Div([
                                    html.H3('Select cryptocurrencies:'),
                                    dcc.Dropdown(
                                        # add an ID to the input box
                                        id='dropdown',
                                        options=options,
                                        # sets a default value
                                        value=['Bitcoin', 'Ethereum', 'Litecoin'],
                                        multi=True
                                    )
                                ], className = 'overlay')
                            ], className = 'half-column'
                        ),
                        # Add second column
                        html.Div(
                            [
                                # Add DatePickerRange
                                html.Div([
                                    html.H3('Select start and end dates:'),
                                    dcc.DatePickerRange(
                                        id='date-picker',
                                        min_date_allowed=datetime(2016, 3, 18),
                                        max_date_allowed=datetime(2021, 4, 30),
                                        start_date=datetime(2020, 5, 1),
                                        end_date=datetime(2021, 4, 30),
                                        stay_open_on_select = True,
                                    )
                                ], className = 'overlay')
                            ], className = 'half-column'
                        )
                    ], className = 'column-wrapper'
                )
            ], className = 'header'
        ),
        html.Div(
            # Add graph inside Loading component
            dcc.Loading(dcc.Graph(id='main-graph')), className = 'graph-wrapper'
        ),
        html.Div(html.P(disclaimer, id = 'disclaimer'))
    ], className = 'dashboard-wrapper')

# Add the server clause
if __name__ == '__main__':
    app.run_server(debug = True)

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.figure_factory as ff

import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

# set the max columns to none
pd.set_option('display.max_columns', None)

## PATHS
ansur_male_path = "../data/ansur/ANSUR II MALE Public.csv"
ansur_female_path = "../data/ansur/ANSUR II FEMALE Public.csv"

## Dataframe
ansur_male = pd.read_csv(ansur_male_path, encoding = 'cp1252')
ansur_female = pd.read_csv(ansur_female_path, encoding = 'cp1252')
df = pd.concat([ansur_male, ansur_female])
drop_list_nonnumeric = ["Date", "Installation", "Component","PrimaryMOS"]
df.drop(drop_list_nonnumeric, axis=1, inplace = True)

## Data Cleaning
NaN_list =[]
for columns in df.columns:
    if df[columns].isnull().sum() > 0:
        print("{name} = {qty}".format(name = columns, qty = df[columns].isnull().sum()))
        NaN_list.append(columns)
        
df = df.drop(NaN_list, axis=1)
df.drop("SubjectNumericRace", axis = 1, inplace = True)

# APP
app = Dash(__name__)

colors = {
    'background': '#CAD7DA',
    'text': '#000000'
}

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    
    # Title
    html.H1(
        children='ANSUR II - Data Exploration',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    
    # Description
    html.Div(children='This app allows you to look at the distribution of the variables in the ansur II dataset.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    # Drop Down
    html.Div([
        html.Div([
            dcc.Dropdown(
                ["Female", "Male", "Both"],
                'Female',
                id='gender'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                df.columns,
                df.columns[0],
                id='variable'
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    # Graph
    dcc.Graph(id='graph'),
    
    # Percentile Slider
    dcc.Slider(
        1,
        99,
        step=None,
        id='percentile',
        value=50,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div(id='output-percentile', style={'backgroundColor': "#FFA500"}),
    
    # Value Slider
    dcc.Slider(
        1,
        99,
        step=None,
        id='value_slider',
        value=50,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div(id='output-value', style={'backgroundColor': "#008B8B"}),
    
    html.Div(children='Summary Stats', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
#     html.Div([
#         html.H4(children='ANSUR II (2012)'),
#         generate_table(df)
#     ])
])

# Callback for updating the graph
@app.callback(
    Output('graph', 'figure'),
    Input('gender', 'value'),
    Input('variable', 'value'),
    Input('percentile', 'value'))
def update_graph(gender, variable, percentile):
    if gender == "Both":
#        dff = df.copy()
#         fig = px.histogram(dff, x=variable, color="Gender",
#                    marginal="box")
        hist_data = [df[df['Gender'] == "Male"][variable], df[df['Gender'] == "Female"][variable]]
        group_labels = ["Male", "Female"]
        fig = ff.create_distplot(hist_data, group_labels, colors=['blue', 'red'])
        
        # plot the vertical line
        fig.add_vline(x=df[variable].quantile(percentile*0.01), line_width=3, line_dash="dash", line_color="green")
        
    else:
        dff = df[df['Gender'] == gender]
#         fig = px.histogram(dff, x=variable,
#                    marginal="box")
        hist_data = [dff[variable]]
        group_labels = [variable]
        if gender == "Male":
            fig = ff.create_distplot(hist_data, group_labels, colors=["blue"])
        else:
            fig = ff.create_distplot(hist_data, group_labels, colors=["red"])
        
        # plot the vertical line
        fig.add_vline(x=dff[variable].quantile(percentile*0.01), line_width=3, line_dash="dash", line_color="green")
        
    fig.update_xaxes(title=variable)
    fig.update_yaxes(title="density")
    
    # Color
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return fig

# Callback for Value Slider
@app.callback([Output(component_id='value_slider', component_property='min'),
               Output(component_id='value_slider', component_property='max'),
              Output(component_id='value_slider', component_property='value')],
              [Input('gender', 'value'),
               Input('variable', 'value')]) # This is the dropdown for selecting variable
def update_value_slider(gender, variable):
    if gender == "Both":
        dff = df.copy()
    else:
        dff = df[df['Gender'] == gender]

    return dff[variable].min(), dff[variable].max(), (dff[variable].min() + dff[variable].max())/2

# Callback for updating percentile->value output
@app.callback(
    Output(component_id='output-percentile', component_property='children'),
    Input('gender', 'value'),
    Input('variable', 'value'),
    Input('percentile', 'value')
)
def update_percentile_div(gender, variable, percentile):
    if gender == "Both":
        dff = df.copy()
    else:
        dff = df[df['Gender'] == gender]
    return f'Percentile -> Value: {dff[variable].quantile(percentile*0.01)} (mm or kg)'

# Callback for updating value->percentile output
@app.callback(
    Output(component_id='output-value', component_property='children'),
    Input('gender', 'value'),
    Input('variable', 'value'),
    Input('value_slider', 'value'),
)
def update_value_div(gender, variable, value):
    if gender == "Both":
        dff = df.copy()
    else:
        dff = df[df['Gender'] == gender]
    return f'Value -> Percentile: {round(scipy.stats.percentileofscore(dff[variable], value), 2)}%'
    
if __name__ == '__main__':
    app.run_server(debug=True)
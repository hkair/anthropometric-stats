# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache
from dash_table import DataTable

import boto3 
import io
from PIL import Image

import os
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# set the max columns to none
pd.set_option('display.max_columns', None)

# AWS S3 Bucket
session = boto3.Session()
s3 = session.resource('s3')

BUCKET_NAME = 'aws-s3-anthropometric-stats'
img_folder = 'body_measurement_images/'

## PATHS
ansur_male_path = "./data/ansur/ANSUR II MALE Public.csv"
ansur_female_path = "./data/ansur/ANSUR II FEMALE Public.csv"

## Dataframe
ansur_male = pd.read_csv(ansur_male_path, encoding = 'cp1252')
ansur_female = pd.read_csv(ansur_female_path, encoding = 'cp1252')
df = pd.concat([ansur_male, ansur_female])
drop_list_nonnumeric = ["Date", "Installation", "Component","PrimaryMOS"]
df.drop(drop_list_nonnumeric, axis=1, inplace = True)

## Data Cleaning
NaN_list = []
for columns in df.columns:
    if df[columns].isnull().sum() > 0:
        print("{name} = {qty}".format(name = columns, qty = df[columns].isnull().sum()))
        NaN_list.append(columns)
        
df = df.drop(NaN_list, axis=1)
df.drop("SubjectNumericRace", axis = 1, inplace = True)

# Race 
race_code = {"White":1, "Black":2, "Hispanic":3, "Asian":4, "Native American":5, "Pacific Islander":6, "Other":8}
races = (["All", "White", "Black", "Hispanic", "Asian", "Native American", "Pacific Islander", "Other"])

# Height
heights = ["All"]
heights.extend([str(55+i)+"-"+str(55+i+1) for i in range(35)])

# Body Measurements - For Proportionality Constants
body_measurements = ['abdominalextensiondepthsitting',
    'acromialheight',
    'acromionradialelength',
    'anklecircumference',
    'axillaheight',
    'balloffootcircumference',
    'balloffootlength',
    'biacromialbreadth',
    'bicepscircumferenceflexed',
    'bicristalbreadth',
    'bideltoidbreadth',
    'bimalleolarbreadth',
    'bitragionchinarc',
    'bitragionsubmandibulararc',
    'bizygomaticbreadth',
    'buttockcircumference',
    'buttockdepth',
    'buttockheight',
    'buttockkneelength',
    'buttockpopliteallength',
    'calfcircumference',
    'cervicaleheight',
    'chestbreadth',
    'chestcircumference',
    'chestdepth',
    'chestheight',
    'crotchheight',
    'crotchlengthomphalion',
    'crotchlengthposterioromphalion',
    'earbreadth',
    'earlength',
    'earprotrusion',
    'elbowrestheight',
    'eyeheightsitting',
    'footbreadthhorizontal',
    'footlength',
    'forearmcenterofgriplength',
    'forearmcircumferenceflexed',
    'forearmforearmbreadth',
    'forearmhandlength',
    'functionalleglength',
    'handbreadth',
    'handcircumference',
    'handlength',
    'headbreadth',
    'headcircumference',
    'headlength',
    'heelanklecircumference',
    'heelbreadth',
    'hipbreadth',
    'hipbreadthsitting',
    'iliocristaleheight',
    'interpupillarybreadth',
    'interscyei',
    'interscyeii',
    'kneeheightmidpatella',
    'kneeheightsitting',
    'lateralfemoralepicondyleheight',
    'lateralmalleolusheight',
    'lowerthighcircumference',
    'mentonsellionlength',
    'neckcircumference',
    'neckcircumferencebase',
    'overheadfingertipreachsitting',
    'palmlength',
    'poplitealheight',
    'radialestylionlength',
    'shouldercircumference',
    'shoulderelbowlength',
    'shoulderlength',
    'sittingheight',
    'sleevelengthspinewrist',
    'sleeveoutseam',
    'span',
    'suprasternaleheight',
    'tenthribheight',
    'thighcircumference',
    'thighclearance',
    'thumbtipreach',
    'tibialheight',
    'tragiontopofhead',
    'trochanterionheight',
    'verticaltrunkcircumferenceusa',
    'waistbacklength',
    'waistbreadth',
    'waistcircumference',
    'waistdepth',
    'waistfrontlengthsitting',
    'waistheightomphalion',
    'wristcircumference',
    'wristheight']

# Compute Constants
for col in df.columns:
    if col in body_measurements:
        df[col+"_pconstant"] = df[col]/df["stature"]
            
# Helper Functions
def percentiles_df(df, measure):
    # measure - the measurement or column in the dataframe
    k_percentiles = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 98, 99]
    
    male = df[df['Gender'] == "Male"][measure]
    female = df[df['Gender'] == "Female"][measure]
    
    percentile_df = []
    for k in k_percentiles:
        if k == 1:
            #print(female.quantile(k*0.01), " {k}ST ".format(k=k), male.quantile(k*0.01))
            d = {
                'FEMALES': female.quantile(k*0.01),
                'Percentile':  " {k}ST ".format(k=k),
                'MALES': male.quantile(k*0.01)
            }
        elif k in [2, 3]:
            #print(female.quantile(k*0.01), " {k}ND ".format(k=k), male.quantile(k*0.01))
            d = {
                'FEMALES': female.quantile(k*0.01),
                'Percentile':  " {k}ND ".format(k=k),
                'MALES': male.quantile(k*0.01)
            }
        else:
            #print(female.quantile(k*0.01), " {k}TH ".format(k=k), male.quantile(k*0.01))
            d = {
                'FEMALES': female.quantile(k*0.01),
                'Percentile':  " {k}TH ".format(k=k),
                'MALES': male.quantile(k*0.01)
            }
        percentile_df.append(d)
    return pd.DataFrame(percentile_df)

def frequency_table(df, measure):
    
    iw = (df[measure].max()-df[measure].min())/10
    data = df[measure]
    n = len(data)
    
    # the number of bins is based on the frequency tables in the original ANSUR II dataset summary
    frequency, intervals = np.histogram(data, bins = 40)
    
    freq = pd.DataFrame(index = np.linspace(1,40,40), columns = ['start', 'end', 'F'])
    # Assign the intervals
    freq['start'] = intervals[:-1]
    freq['end'] = intervals[1:]
    # Assing Absolute frecuency
    freq['F'] = frequency
    freq["Fpct"] = freq['F']/n
    freq["CumF"] = freq['F'].cumsum()
    freq["CumFPct"] = freq['CumF']/n
    
    return freq

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img
    
# APP
app = Dash(__name__)
server = app.server

colors = {
    'background': '#CAD7DA',
    'text': '#000000'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    
    # Title
    html.H1(
        children='ANSUR II - Data Exploration & Body Proportions',
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
    
    html.A("Link to ANSUR II: Methods and Summary Statistics", href='http://tools.openlab.psu.edu/publicData/ANSURII-TR15-007.pdf', target="_blank", 
    style={
        'textAlign': 'center'
    }),

    # Drop Down
    html.Div([
        html.Div([
            dcc.Dropdown(
                ["Female", "Male", "Both"],
                'Both',
                id='gender'
            )
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                races,
                'All',
                id='race'
            )
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                heights,
                'All',
                id='height'
            )
        ], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                df.columns,
                df.columns[0],
                id='variable'
            )
        ], style={'width': '25%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    # Images
    dcc.Graph(id='image'),
    
    # Variable Description
    html.Div(id="description",
             children='This app allows you to look at the distribution of the variables in the ansur II dataset.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    # Distribution Graph
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
    
    # Summary Stats
    html.Div(children='Summary Stats', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div(id='mean', children='MEAN: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='std-error', children='STD ERROR (MEAN): ',style={'backgroundColor': "#99E863"}),
    html.Div(id='std-deviation', children='STANDARD DEVIATION: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='std-error-2', children='STD ERROR (STD DEV): ', style={'backgroundColor': "#99E863"}),
    html.Div(id='min', children='MIN: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='max', children='MAX: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='skewness', children='SKEWNESS: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='kurtosis', children='KURTOSIS: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='coefficient', children='COEFFICIENT OF VARIATION: ', style={'backgroundColor': "#99E863"}),
    html.Div(id='participants', children='NUMBER OF PARTICIPANTS: ', style={'backgroundColor': "#99E863"}),
    
    html.Div(children='Percentile', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    DataTable(
        id='table',
        data=[]
    ),
    
    html.Div(children='Frequency Table', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    # Frequency Plot
    DataTable(
        id='frequency',
        data=[]
    ),
    
    # Correlation Matrix
    html.Div(children='Correlation Matrix', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(id="correlation-graph"),
    html.P("Variables: "),
    dcc.Checklist(
        id='correlation-variables',
        options=df.columns.values,
        value=["stature", "Weightlbs", "abdominalextensiondepthsitting", "biacromialbreadth", "waistcircumference", "sittingheight"],
    ),
    
    # Pie Chart
    html.Div(children='Pie Chart', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Dropdown(
        ['DODRace', 'Age', 'Branch', 'SubjectsBirthLocation'],
        'DODRace',
        id='names'
    ),
    
    dcc.Graph(id="pie-chart"),
])

### CALLBACKS

# Callback for updating images
@app.callback(
    Output("image", 'figure'),
    Input('variable', 'value')
)
def update_images(variable):
    bucket = s3.Bucket(BUCKET_NAME)
    
    # for proportionality constants
    if '_pconstant' in variable:
        variable = variable.replace('_pconstant','')
    
    object_names = []
    for bucket_object in bucket.objects.all():
        if variable in bucket_object.key:
            object_names.append(bucket_object.key)
    
    output = None
    for i, key in enumerate(object_names):
        obj = bucket.Object(key)
        response = obj.get()
        file_stream = response['Body']
        im = Image.open(file_stream)
        np_img = np.array(im)
        if i == 0:
            output = np_img
        else:
            output = concat_images(output, np_img)
            
    fig = px.imshow(output)
    
    return fig

# Callback for updating variable description
@app.callback(
    Output('description', 'children'),
    Input('variable', 'value')
)
def update_description(variable):
    description = ""
    
    # for proportionality constants
    if '_pconstant' in variable:
        variable = variable.replace('_pconstant','')
    
    # read from description.txt
    myfile = open("./description.txt", encoding='utf8')
    while myfile:
        line  = myfile.readline()
        if variable in line:
            print(line)
            i = 0
            while line != "\n":
                line = myfile.readline()
                description += line
                if i == 0:
                    description += ": "
                description += "\n"
                i+=1
            break
    myfile.close() 
    return description
    
# Callback for updating the graph
@app.callback(
    Output('graph', 'figure'),
    Input('gender', 'value'),
    Input('race', 'value'),
    Input('height', 'value'),
    Input('variable', 'value'),
    Input('percentile', 'value'))
def update_graph(gender, race, height, variable, percentile):
    # Gender
    if gender == "Both":
        # Race
        if race != "All":
            dff = df[df['DODRace'] == race_code[race]]
        else:
            dff = df.copy()
        
        # Height
        if height != "All":
            h1 = int(height[:2])
            h2 = int(height[-2:])
            dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] <= h2)]
            
        hist_data = [dff[dff['Gender'] == "Male"][variable], dff[dff['Gender'] == "Female"][variable]]
        group_labels = ["Male", "Female"]
        fig = ff.create_distplot(hist_data, group_labels, colors=['blue', 'red'])
        
        # plot the vertical line
        fig.add_vline(x=dff[variable].quantile(percentile*0.01), line_width=3, line_dash="dash", line_color="green")
        
    else:
        if race != "All":
            dff = df[(df['Gender'] == gender) & (df['DODRace'] == race_code[race])]
        else:
            dff = df[(df['Gender'] == gender)]
            
        # Height
        if height != "All":
            h1 = int(height[:2])
            h2 = int(height[-2:])
            dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] <= h2)]
            
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

# Callback for Summary Stats
@app.callback(
    [Output('mean', 'children'), 
     Output('std-error', 'children'), 
     Output('std-deviation', 'children'), 
     Output('std-error-2', 'children'), 
     Output('min', 'children'), 
     Output('max', 'children'), 
     Output('skewness', 'children'), 
     Output('kurtosis', 'children'), 
     Output('coefficient', 'children'), 
     Output('participants', 'children')],
    Input('gender', 'value'),
    Input('race', 'value'),
    Input('height', 'value'),
    Input('variable', 'value'))
def update_summary(gender, race, height, variable):
        
    # Race
    if race != "All":
        dff = df[df['DODRace'] == race_code[race]]
    else:
        dff = df.copy()
    
    # Gender
    if gender != "Both":
        dff = dff[dff['Gender'] == gender]

    # Height
    if height != "All":
        h1 = int(height[:2])
        h2 = int(height[-2:])
        dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] <= h2)]
    
    # Measure 
    dff = dff[variable]
    
    return "MEAN: " + str(np.mean(dff)), \
           "STD ERROR (MEAN): " + str(scipy.stats.sem(dff)), \
           "STANDARD DEVIATION: " + str(np.std(dff)), \
           "STD ERROR (STD DEV): " + str(np.sqrt(np.sqrt(2*np.power(np.std(dff), 4)/(len(dff)-1)))), \
           "MIN: " + str(np.min(dff)), \
           "MAX: " + str(np.max(dff)), \
           "SKEWNESS: " + str(scipy.stats.skew(dff)), \
           "KURTOSIS: " + str(scipy.stats.kurtosis(dff, fisher=False)), \
           "COEFFICIENT OF VARIATION: " + str(scipy.stats.variation(dff)), \
           "NUMBER OF PARTICIPANTS: " + str(len(dff))

# Callback for Summary Stats - Percentile
@app.callback(
    [Output("table", "data"), Output('table', 'columns')],
    [Input('race', 'value'),
    Input('height', 'value'),
    Input('variable', 'value')]
)
def updateTable(race, height, variable):
    # Race
    if race != "All":
        dff = df[df['DODRace'] == race_code[race]]
    else:
        dff = df.copy()

    # Height
    if height != "All":
        h1 = int(height[:2])
        h2 = int(height[-2:])
        dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] <= h2)]
    
    dff = percentiles_df(dff, variable)
    return dff.to_dict('records'), tuple([ {'id': p, 'name': p} for p in dff.columns])

# Callback for Frequency Table
@app.callback(
    [Output("frequency", "data"), Output('frequency', 'columns')],
    [Input('race', 'value'),
    Input('height', 'value'),
    Input('variable', 'value')]
)
def updateFrequency(race, height, variable):
    # Race
    if race != "All":
        dff = df[df['DODRace'] == race_code[race]]
    else:
        dff = df.copy()

    # Height
    if height != "All":
        h1 = int(height[:2])
        h2 = int(height[-2:])
        dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] <= h2)]
    
    dff = frequency_table(dff, variable)
    return dff.to_dict('records'), tuple([ {'id': p, 'name': p} for p in dff.columns])

    
# Callback for updating Correlation Matrix
@app.callback(
    Output("correlation-graph", "figure"), 
    Input("correlation-variables", "value"))
def filter_heatmap(cols):
    fig = px.imshow(df[cols].corr())
    # Color
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig

# Callback for updating Pie Chart
@app.callback(
    Output("pie-chart", "figure"), 
    Input("names", "value"))
def generate_chart(names):
    labels = df[names].value_counts().index
    values = df[names].value_counts().values
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial'
                            )])
   
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
               Input('race', 'value'),
               Input('height', 'value'),
               Input('variable', 'value')]) # This is the dropdown for selecting variable
def update_value_slider(gender, race, height, variable):
    if gender == "Both":
        if race != "All":
            dff = df[df['DODRace'] == race_code[race]]
        else:
            dff = df.copy()
    else:
        if race != "All":
            dff = df[(df['Gender'] == gender) & (df['DODRace'] == race_code[race])]
        else:
            dff = df[df['Gender'] == gender]
    
    # Height
    if height != "All":
        h1 = int(height[:2])
        h2 = int(height[-2:])
        dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] < h2)] 

    return dff[variable].min(), dff[variable].max(), (dff[variable].min() + dff[variable].max())/2

# Callback for updating percentile->value output
@app.callback(
    Output(component_id='output-percentile', component_property='children'),
    Input('gender', 'value'),
    Input('race', 'value'),
    Input('height', 'value'),
    Input('variable', 'value'),
    Input('percentile', 'value')
)
def update_percentile_div(gender, race, height, variable, percentile):
    if gender == "Both":
        if race != "All":
            dff = df[df['DODRace'] == race_code[race]]
        else:
            dff = df.copy()
    else:
        if race != "All":
            dff = df[(df['Gender'] == gender) & (df['DODRace'] == race_code[race])]
        else:
            dff = df[df['Gender'] == gender]

    # Height
    if height != "All":
        h1 = int(height[:2])
        h2 = int(height[-2:])
        dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] < h2)] 
        
    return f'Percentile -> Value: {dff[variable].quantile(percentile*0.01)} (mm (inches) or kg (lbs))'

# Callback for updating value->percentile output
@app.callback(
    Output(component_id='output-value', component_property='children'),
    Input('gender', 'value'),
    Input('race', 'value'),
    Input('height', 'value'),
    Input('variable', 'value'),
    Input('value_slider', 'value'),
)
def update_value_div(gender, race, height, variable, value):
    if gender == "Both":
        if race != "All":
            dff = df[df['DODRace'] == race_code[race]]
        else:
            dff = df.copy()
    else:
        if race != "All":
            dff = df[(df['Gender'] == gender) & (df['DODRace'] == race_code[race])]
        else:
            dff = df[df['Gender'] == gender]
            
    # Height
    if height != "All":
        h1 = int(height[:2])
        h2 = int(height[-2:])
        dff = dff[(dff["Heightin"] >= h1) & (dff["Heightin"] < h2)] 
        
    return f'Value -> Percentile: {round(scipy.stats.percentileofscore(dff[variable], value), 2)}%'
    
if __name__ == '__main__':
    app.run_server(debug=True)
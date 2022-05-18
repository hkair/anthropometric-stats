# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, State
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

import cv2 
#from base64 import decodebytes
import base64

from anthropometricProp import BodySkeleton
from poseModule import poseDetector

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
        #print("{name} = {qty}".format(name = columns, qty = df[columns].isnull().sum()))
        NaN_list.append(columns)
        
df = df.drop(NaN_list, axis=1)
df.drop("SubjectNumericRace", axis = 1, inplace = True)

# Race 
race_code = {"White":1, "Black":2, "Hispanic":3, "Asian":4, "Native American":5, "Pacific Islander":6, "Other":8}
races = (["All", "White", "Black", "Hispanic", "Asian", "Native American", "Pacific Islander", "Other"])

# Height
heights = ["All"]
heights.extend([str(150+i) for i in range(50)])

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
def subsetPopulation(df, gender, race, height, variable):
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
        h1 = int(height[:3])*10 - 5
        h2 = int(height[:3])*10 + 5
        dff = dff[(dff["stature"] >= h1) & (dff["stature"] <= h2)]
        
    if variable == "All":
        return dff
    
    return dff[variable]

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

font_sizes = {
    "h1": 30,
    "h2": 20,
    "h3": 15
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
        'color': colors['text'], 
        'font-size': font_sizes["h2"]
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
    
    # Upload Image
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    
    html.Div(
        children=[
            html.Div(id='pose-estimation'),
        ]
    ),
    
    #dcc.Graph(id='pose-estimation'),
    
    html.Div([
        # Images
        html.Div([
            html.H3('Measurement Image'),
            dcc.Graph(id='image'),
        ], style={'width': '49%', 'display': 'inline-block'}),
           
        # Body Proportion 
        html.Div([
            html.H3('Anthropometric Proportionality', style = {'margin':'auto','width': "50%"}),
            html.Div(
                dcc.Graph(id='body-graph'),
                style={'margin':'auto','width': "50%"}
            ),
            html.Div(
                dcc.Dropdown(
                    ["Absolute", "Ratio"],
                    'Absolute',
                    id='proportion'
                ),
                style={'margin':'auto','width': "50%"}
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
    ], className="row"),
    
    # Variable Description
    html.Div(id="description",
             children='This app allows you to look at the distribution of the variables in the ansur II dataset.', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-size': font_sizes["h2"]
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
        'color': colors['text'],
        'font-size': font_sizes["h1"]
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
        'color': colors['text'],
        'font-size': font_sizes["h1"]
    }),
    
    DataTable(
        id='table',
        data=[]
    ),
    
    html.Div(children='Frequency Table', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-size': font_sizes["h1"]
    }),
    
    # Frequency Plot
    DataTable(
        id='frequency',
        data=[]
    ),
    
    # Correlation Matrix
    html.Div(children='Correlation Matrix', style={
        'textAlign': 'center',
        'color': colors['text'],
        'font-size': font_sizes["h1"]
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
        'color': colors['text'],
        'font-size': font_sizes["h1"]
    }),
    dcc.Dropdown(
        ['DODRace', 'Age', 'Branch', 'SubjectsBirthLocation'],
        'DODRace',
        id='names'
    ),
    
    dcc.Graph(id="pie-chart"),
])


### CALLBACKS

def parse_contents(contents, filename, date):

    # Remove 'data:image/png;base64' from the image string,
    # see https://stackoverflow.com/a/26079673/11989081
    content_type, content_string = contents.split(',')
    print(content_type)
    img = np.array(Image.open(io.BytesIO(base64.b64decode(content_string))))

    # Convert the image string to numpy array and create a
    # Plotly figure, see https://plotly.com/python/imshow/
    detector = poseDetector()
    while True:
        img = detector.getPose(img)
        lmList = detector.getPosition(img)
        lengths = detector.getLengths(img)
        img = detector.getHeadBox(img, False)

        img = detector.getProportions(img)
        cv2.imshow('img', img) #display the captured image
        #if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.destroyAllWindows()
        break

    fig = px.imshow(img)

    # Hide the axes and the tooltips
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=20, b=0, l=0, r=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            linewidth=0
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            linewidth=0
        ),
        hovermode=False
    )

    return html.Div([
        html.H5(filename),
        dcc.Graph(
            figure=fig,
            config={'displayModeBar': True} # Always display the modebar
        )
    ])

# Callback for inputing image 
@app.callback(Output('pose-estimation', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_pose(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children            

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
    
    # Color
    fig.update_layout(
        plot_bgcolor='#808080',
        paper_bgcolor='#808080',
        font_color=colors['text']
    )
    
    return fig

@app.callback(
    Output('body-graph', 'figure'),
    Input('gender', 'value'),
    Input('race', 'value'),
    Input('height', 'value'),
    Input('proportion', 'value')
)
def updateBodySkeleton(gender, race, height, proportion):
    body = BodySkeleton(df, height, gender, race, proportion)
    return body.getFig()

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
        dff = subsetPopulation(df, gender, race, height, "All")

        # Handle if no heights within range for one of the genders
        if len(dff[dff['Gender'] == "Male"][variable]) > 1 and len(dff[dff['Gender'] == "Female"][variable]) > 1:
            hist_data = [dff[dff['Gender'] == "Male"][variable], dff[dff['Gender'] == "Female"][variable]]
            group_labels = ["Male", "Female"]
            fig = ff.create_distplot(hist_data, group_labels, colors=['blue', 'red'])
        elif (len(dff[dff['Gender'] == "Male"][variable]) == 0) and len(dff[dff['Gender'] == "Female"][variable]) > 1:
            hist_data = [dff[dff['Gender'] == "Female"][variable]]
            group_labels = ["Female"]
            fig = ff.create_distplot(hist_data, group_labels, colors=['red'])
        elif (len(dff[dff['Gender'] == "Male"][variable]) > 1) and len(dff[dff['Gender'] == "Female"][variable]) == 0:
            hist_data = [dff[dff['Gender'] == "Male"][variable]]
            group_labels = ["Male"]
            fig = ff.create_distplot(hist_data, group_labels, colors=['blue'])
        else:
            fig = go.Figure()
        
        # plot the vertical line
        fig.add_vline(x=dff[variable].quantile(percentile*0.01), line_width=3, line_dash="dash", line_color="green")
        
    else:
        dff = subsetPopulation(df, gender, race, height, variable)
        hist_data = [dff]
        group_labels = [variable]
        
        if len(dff) <= 1 :
            fig = go.Figure()
        elif gender == "Male":
            fig = ff.create_distplot(hist_data, group_labels, colors=["blue"])
        else:
            fig = ff.create_distplot(hist_data, group_labels, colors=["red"])
        
        # plot the vertical line
        fig.add_vline(x=dff.quantile(percentile*0.01), line_width=3, line_dash="dash", line_color="green")
    
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
        
    dff = subsetPopulation(df, gender, race, height, variable)
    
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
        h1 = int(height[:3])*10 - 5
        h2 = int(height[:3])*10 + 5
        dff = dff[(dff["stature"] >= h1) & (dff["stature"] <= h2)]
    
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
        h1 = int(height[:3])*10 - 50
        h2 = int(height[:3])*10 + 50
        dff = dff[(dff["stature"] >= h1) & (dff["stature"] <= h2)]
    
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
    # Flip the race_code dictionary key to values and values to keys
    code_race = { val:key for key, val in race_code.items() }
    
    # if variable is DODRace, use the int:race mapping
    if names == "DODRace":
        labels = [ code_race[i] for i in df[names].value_counts().index]
    else:
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
    dff = subsetPopulation(df, gender, race, height, variable)

    return dff.min(), dff.max(), (dff.min() + dff.max())/2

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
    dff = subsetPopulation(df, gender, race, height, variable)
    
    if len(dff) > 0:
        return f'Percentile -> Value: {dff.quantile(percentile*0.01)} (mm (inches) or kg (lbs))'
    return "Invalid: No population exists in the Dataset"

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
    dff = subsetPopulation(df, gender, race, height, variable)
    
    # Handle cases where subset population doesn't exist
    if len(dff) > 0:
        return f'Value -> Percentile: {round(scipy.stats.percentileofscore(dff, value), 2)}%'
    return "Invalid: No population exists in the Dataset"
    
if __name__ == '__main__':
    app.run_server(debug=True)
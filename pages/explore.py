from __future__ import annotations

import dash
# from dash_bootstrap_templates import load_figure_template


dash.register_page(
    __name__,
    path='/explore',
    title="Customer Churn",
)

# load_figure_template("minty")

import pathlib
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight') # default 

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

def wrangle(file):
    df = pd.read_csv(file)
    df["PhoneLine"] = df['MultipleLines'].replace({'No phone service':'No','No':'1','Yes':'>1'})
    df['TotalCharges'] = df['TotalCharges'].replace({' ':'0.0'}).astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({0:'No',1:"Yes"}).astype(str)
    df['churn1'] =df['Churn'].replace({"Yes":1,"No":0})
    df['TenureThreshold'] = pd.qcut(df['tenure'], 4, labels=["<= 9 months", "10-29 months", "30-55 months",">55 months"])
    df.replace({'No internet service':'No I-S',
                'Bank transfer (automatic)':'Bank transfer',
                'Credit card (automatic)':'Credit card',
                'No phone service':'No P-S' },inplace=True)
    
    # df.columns=df.columns.title()
    return df

# rangeslider_marks = {x:x}

df = wrangle(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))

lab_var = {x:' '.join(re.findall('[a-zA-Z][^A-Z]*', x)).title()  for x in df.columns[1:]}


layout = dbc.Container(
    [   
        
        dbc.Row(
            [
                dbc.Col([

                    html.H2(
                        "Telecom Customer Churn : Customers Data",
                        style={"font-weight": "bold","color": "#0084d6"},
                        className="text-center mb-4",
                    ),

                    # html.H6(
                    #     "Profile",
                    #     className="text-center",#text-info mb-4
                    #     style={"margin-buttom": "5px"},
                    # ),
                    # # width=6,
                ]
                ),
            ]
        ),

        
        html.Div(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Customer ID",style={"color": "#0084d6","font-weight": "bold","text-align": "center"},),
                                        html.Hr(),
                                        html.Div(
                                        [
                                            dcc.RadioItems(id="CustomerID",options=df['customerID'],value= df['customerID'][2],
                                                labelStyle={"display": "block","text-align": "justify",} ,
                                                inputStyle={"color": "green",})
                                        ],
                                        style={"maxHeight": "300px", "overflow": "scroll",'margin-left':'35px'}
                                        
                                    )],
                                ),
                                # className="card-header",
                                id="cardID0", ),
                            ],
                            style={'margin-right':'5px'},
                        ),
                        dbc.Row(
                            [dbc.Card(
                                dbc.CardBody(html.Div(id="churnStatus")),
                                # className="card-header",
                                id="cardID1", ),
                            ],
                            style={'margin-right':'5px','margin-top':'10px'},
                        ),
                    ],
                    className="col-md-12 col-lg-3  card-chart-contain",
                    # style={'margin-left':'15px'},
                ),
                dbc.Col(
                    [dbc.Card(
                        dbc.CardBody(
                            [
                            html.H4("Demographics",style={"color": "#0084d6","font-weight": "bold","text-align": "center"},),
                            html.Hr(),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                        html.Div(
                                            [html.P(str(lab_var[i]+" :"),style={"color": "#909090","font-weight": "bold"},),],
                                            style={'display':'inline-block'},
                                        ),
                                        # html.Br(),
                                        html.Div(
                                            [html.P(id=str(i)+"_info",style={"color": "#0084d6","font-weight": "bold","font-size":'90%'},),],
                                            style={'display':'inline-block'}
                                        ),


                                        ],
                                        style={'display':'flex-justify'},    
                                        className="mb-4",

                                    ) for i in df.columns[1:5]

                                ],
                                style={"height": "430px",},
                                id="DemographicInfo"
                            )
                            ]
                        ),
                        # className="card-header",
                        id="cardID2",
                    ),
                    ],
                    className="col-md-12 col-lg-3  card-chart-contain",
                ),
                dbc.Col(
                    [dbc.Card(
                        dbc.CardBody(
                            [
                            html.H4("Subscriptions",style={"color": "#0084d6","font-weight": "bold","text-align": "center"},),
                            html.Hr(),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                        html.Div(
                                            [html.P(lab_var[i]+" :",style={"color": "#909090","font-weight": "bold","text-align": "justify"},),],
                                            style={'display':'inline-block'}
                                        ),
                                        # html.Br(),
                                        html.Div(
                                            [html.P(id=str(i)+"_info",style={"color": "#0084d6","font-weight": "bold","font-size":'90%'},),],
                                            style={'display':'inline-block'}
                                        ),


                                        ],
                                        style={'display':'flex-justify'},    
                                        className="mb-4",

                                    ) for i in df.columns[6:15]

                                ],
                                style={"height": "430px",},
                                id="ServicesInfo"
                            )
                            ]
                        )
                        ,id="cardID3",
                    ),
                    ],
                    className="col-md-12 col-lg-3  card-chart-contain",
                ),

                dbc.Col(
                    [dbc.Card(
                        dbc.CardBody(
                            [
                            html.H4("Billings Details",style={"color": "#0084d6","font-weight": "bold","text-align": "center"},),
                            html.Hr(),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                        html.Div(
                                            [html.P(lab_var[i]+" :",style={"color": "#909090","font-weight": "bold","text-align": "justify"},),],
                                            style={'display':'inline-block'}
                                        ),
                                        # html.Br(),
                                        html.Div(
                                            [html.P(id=str(i)+"_info",style={"color": "#0084d6","font-weight": "bold","font-size":'90%'},),],
                                            style={'display':'inline-block',}
                                        ),


                                        ],
                                        # style={'display':'flex-justify'},    
                                        className="mb-4",

                                    ) for i in sorted(['tenure']+list(df.columns[15:20]))

                                ],
                                style={"height": "430px",},
                                id="FacturationsInfo"
                            )
                            ]
                        )
                        ,id="cardID4",
                    ),
                    ],
                    className="col-md-12 col-lg-3  card-chart-contain",
                ),
                
            ],
            className="row flex-display",

        ), 
                          
       
       
    ]
)

@callback(
    # [Output("Attribute", "style")]+
   [Output("cardID"+str(i), "style") for i in range(5)],
   Input("color_bg", 'data'),
)

def toggle_theme(data):
    th = [{'background-color': str(data)}]*5
    return th



@callback(
    [Output("churnStatus", "children")]+
    [Output(str(i)+"_info", "children") for i in df.columns[1:5]]+
    [Output(str(i)+"_info", "children") for i in df.columns[6:15]]+
    [Output(str(i)+"_info", "children") for i in sorted(['tenure']+list(df.columns[15:20]))],
   Input("CustomerID", 'value'),
)

def UpdateStatus(value):
    X = df[df["customerID"]==str(value)].copy()
    cond =  X['Churn'].unique()[0]=="Yes"
    # print( X['Churn'].unique()[0]=="No")
    color = 'red' if cond else 'green'
    name = 'Churned' if cond else 'Stay'
    status = html.P(name,style={"color": color,"font-weight": "bold","font-size": "300%","text-align": "center"},)
    gender = [html.I(className="menu-icon tf-icons bx bx-"+str(X['gender'].unique()[0]).lower()), X['gender'].unique()[0]]
    senior = X['SeniorCitizen'].unique()[0]
    partner = X['Partner'].unique()[0]
    dependents = X['Dependents'].unique()[0]
    services = [X[i].unique()[0] for i in df.columns[6:15]]
    facturaction = [X[i].unique()[0] for i in sorted(['tenure']+list(df.columns[15:20]))]
    liste = [status,gender,senior,partner,dependents]+services+facturaction
    return liste

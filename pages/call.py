from __future__ import annotations

import dash
from dash_bootstrap_templates import load_figure_template

# dash.register_page(__name__, path='/', name='EDA')
dash.register_page(
    __name__,
    path='/call',
    title="Customer Churn",
)

# load_figure_template("minty")
from dash import dcc
from dash import html
import pathlib
import re
import math
import numpy as np
import pandas as pd
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, Input, Output, callback, dcc, html,State
from plotly.subplots import make_subplots
from dash import dash_table

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

#  axis formatting
axis_font = dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        zeroline=False,
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='San-Serif',
            size=12,
            # color='black',
        ),
    )

def find_row_in_df(row, df):
    for ix in df.index:
        if np.all(df.iloc[ix].values == row):
            return ix

    return -1

def NamedGroup(children, label, **kwargs):
    return html.Div(
        [
            html.P(label,
                style={
                    "font-weight": "bold",
                    },),
            # html.Hr(),
            children,
            html.Hr(),
        ],
        **kwargs
        # style={"font-size": ".9vw",},
    )
    

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

df = wrangle(DATA_PATH.joinpath('telco-customer-churn-by-IBM.csv'))
# lab_var = {x:' '.join(re.findall('[a-zA-Z][^A-Z]*', x)).title()  for x in df.columns[1:]}
lab_var = {x:' '.join(re.findall('[a-zA-Z][^A-Z]*', x)).title()  for x in df.columns[1:]}
qual_var=[x for x in df.columns[1:-3] if df[x].dtype=='O' or df[x].nunique()<5]
quant_var = [x for x in df.columns if df[x].dtype!='O' and df[x].nunique()>5]
# var = df.columns[1:]
Personals = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
Services = [ 'DeviceProtection','OnlineBackup','TechSupport',
            'StreamingTV','OnlineSecurity', 'StreamingMovies']
Contracts = ['Contract', 'PaperlessBilling','PaymentMethod',]




layout = html.Div(
    [   
        

        dcc.Store(id="data_frame"),
       
        html.Div(
            [
                
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2(
                                    "Telecom Customer Churn : Overview of Customers Behavior",
                                    style={"font-weight": "bold","color": "#0084d6",},
                                    className="text-center mb-4",

                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Get insight into the Data",
                                                    style={"margin-top": "0px",},
                                                ),
                                            ],
                                            style={'display':'inline-block'} 
                                             
                                        ),
                                        html.Div(
                                            [
                                                html.A(
                                                    html.Img(
                                                        id='info-btn',
                                                        src="./assets/images/info.png",
                                                        height="25px",
                                                        style={"margin-top": "-5px"},
                                                    ),
                                                ),
                                            ],
                                            style={'display':'inline-block'} 
                                        ),
                                    ],
                                    style={'display':'flex-center'},    
                                    className="text-center mb-4",

                                ),
                                dbc.Tooltip("Information", target="info-btn",placement='right'),

                                dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Page Information")),
                                    dbc.ModalBody(
                                        dcc.Markdown('''
                                        This page gives a general information about the `services purcharse`, `churn rate trend`, `average lifespan` and `revenu generated` 
                                        this months base on customers profile.Understanding customers behavior base on their profile can be investigated with more detail by using the filters and color variable available.
                                        

                                        `What to look at` 😉


                                        "It's interesting to note that[ churn customers] tend to prefer `month-to-month` contracts,
                                         with an average lifespan of `14 months`. They also tend to use more `fiber optic` Internet service,
                                          subscribe to more streaming services (`TV & movies`), and prefer `electronic checks` as a payment method.
                                           Our data shows that there is no significant difference between [genders] in this regard.
                                            If you'd like to see these trends more clearly, you can filter the data to show only churn customers and play with the color variable."
                                        
                                        ''')
                                    ),
                                ],
                                id="info-modal",
                                size="lg",
                                is_open=False,
                                # contentClassName='modalcontent1',
                                ),
                            ],
                        ),
                    ],
                    # className="three column",
                    id="title",
                ),
                html.Div(
                    # create empty div for align cente
                    # className="one-third column",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
       
        html.Div(
            [

                html.Div(
                    [   
                        html.Div(
                            [

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(id="cases-text"),
                                                html.P(
                                                    "Cases",
                                                    style={
                                                        "text-align": "center",
                                                        "color": "#909090",
                                                        "font-weight": "bold",
                                                    },
                                                ),
                                                html.Br(),
                                                html.P(
                                                    id="case-number",
                                                    style={
                                                        "text-align": "center",
                                                        # # "color": "#EC1E3D",
                                                        "font-weight": "bold",
                                                        "font-size": 30
                                                    },
                                                ),
                                                html.Br(),
                                            ],
                                            className="card-header",
                                            # style={"background-color":"#fff"},
                                            id="bg_id1",
                                        ),

                                        html.Div(
                                            [
                                                html.P(id="churn-text"),
                                                html.P(
                                                    "Churn",
                                                    style={
                                                        "text-align": "center",
                                                        "color": "#909090",
                                                        "font-weight": "bold",
                                                    },
                                                ),
                                                html.P(
                                                    id="churn-number",
                                                    style={
                                                        "text-align": "center",   
                                                        "color": "#EC1E3D",
                                                        "font-weight": "bold",
                                                        "font-size": 30
                                                    },
                                                ),
                                                 html.P(
                                                    "Churn rate",
                                                    style={
                                                        "text-align": "center",
                                                        "color": "#909090",
                                                        "font-weight": "bold",
                                                    },
                                                ),
                                                html.P(
                                                    id="churn-rate",
                                                    style={
                                                        "text-align": "center",   
                                                        "color": "#EC1E3D",
                                                        "font-weight": "bold",
                                                        "font-size": 15,
                                                    },
                                                )
                                            ],
                                            id="bg_id2",
                                            className="card-header",
                                        ),
                                

                                    ],
                                    id="right-column",
                                    className="col-md-12 col-lg-3 mb-md-0 mb-4 card-chart-container",
                                    style={'margin-left':'15px','margin-right':'-15px'},
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(id="c-chart",figure={
                                                    "layout": {"title": "Contract Chart",
                                                    "height": 280,
                                                    }}
                                                )
                                            ],
                                            className="card-head",
                                            id="bg_id3",
                                        )
                                    ],
                                    className="col-md-12 col-lg-9 mb-md-0 mb-4 card-chart-contain",
                                ),
                        
                            ],
                            className="row flex-display",
                        ),

                      

                        html.Div(
                            [dcc.Graph(id="service-chart",figure={
                                        "layout": {"title": "Service Chart",
                                        "height": 340, }})
                            ],
                            className="card-head",
                            id="bg_id4",
                            
                        ),
                        

                        html.Div(
                            [dcc.Graph(id="rate-chart",figure={
                                        "layout": {"title": "rate Chart",
                                        "height": 320, }})
                            ],
                            id="bg_id5",
                            className="card-head",
                            
                        ), 
                    ],
                    id="right-column",
                    className="col-md-12 col-lg-9 mb-md-0 mb-4 card-chart-container",
                ),
            
                html.Div(
                    [   
                        
                    html.Div([
                        # html.Div(
                        #     [
                        #         dbc.Carousel(
                        #             items=[

                        #                 {"key": "1", "src": "/assets/ch.png"},
                        #                 {"key": "2", "src": "/assets/ch2.png"},
                        #                 {"key": "3", "src": "/assets/ch3.png"},
                        #                 ],
                        #             interval=10000,
                        #             className="carousel-fade",
                        #         )
                        #     ],
                        #     # className="card-header",
                        #     ),
                       

                        html.Div(
                            [dcc.Dropdown(
                                id="Variable",
                                clearable=False,
                                searchable=False,
                                multi=False,
                                placeholder="Choose Color",
                                options=[
                                    {"label": lab_var[c], "value": c,}
                                    for c in Contracts+Personals
                                ],
                                value="Contract",
                                # style={'background-color': '#222430','color':'#0a66c2'},
                                )
                            ],
                            # className="card-header",
                            # id='wells'
                            style={"color":"#0a66c2",},
                        ),
                        html.Hr(),

                        html.Div(
                            [ 
                            dbc.Row(
                                [
                                    dbc.Button(
                                        "Clear Filter",  outline=True, color="primary",id="clear-button", className="me-1",size="sm", n_clicks=0,disabled=True
                                    ),
                                ],
                            ),
                            html.Hr(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(
                                            dbc.Checklist( 
                                                    id= 'gender',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['gender'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                ),
                                                label=lab_var['gender'],
                                                # id="control-item-gender",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id='Dependents',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['Dependents'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['Dependents'],
                                                # id="control-item-Dependents",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'Partner',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['Partner'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['Partner'],
                                                # id="control-item-Partner",
                                            )) ,
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'SeniorCitizen',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['SeniorCitizen'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['SeniorCitizen'],
                                                # id="control-item-SeniorCitizen",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'PhoneService',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['PhoneService'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['PhoneService'],
                                                # id="control-item-PhoneService",
                                            )) ,
        
                                ]
                            ),   
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'MultipleLines',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['MultipleLines'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['MultipleLines'],
                                                # id="control-item-MultipleLines",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'DeviceProtection',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['DeviceProtection'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['DeviceProtection'],
                                                # id="control-item-DeviceProtection",
                                            )) ,
        
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'OnlineBackup',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['OnlineBackup'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['OnlineBackup'],
                                                # id="control-item-OnlineBackup",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'TechSupport',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['TechSupport'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['TechSupport'],
                                                # id="control-item-TechSupport",
                                            )) ,
        
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'StreamingMovies',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['StreamingMovies'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['StreamingMovies'],
                                                # id="control-item-StreamingMovies",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'StreamingTV',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['StreamingTV'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['StreamingTV'],
                                                # id="control-item-StreamingTV",
                                            )) ,
        
                                ]
                            ), 
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'OnlineSecurity',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['OnlineSecurity'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['OnlineSecurity'],
                                                # id="control-item-OnlineSecurity",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'InternetService',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['InternetService'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['InternetService'],
                                                # id="control-item-InternetService",
                                            )) ,
        
                                ]
                            ), 
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'Contract',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['Contract'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['Contract'],
                                                # id="control-item-Contract",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'PaymentMethod',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['PaymentMethod'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['PaymentMethod'],
                                                # id="control-item-PaymentMethod",
                                            )) ,
        
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        NamedGroup(dbc.Checklist( 
                                                    id= 'Churn',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['Churn'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block",} ,
                                                )
                                                ,
                                                label=lab_var['Churn'],
                                                # id="control-item-Churn",
                                            )) ,
                                    dbc.Col(
                                        NamedGroup( dbc.Checklist( 
                                                    id= 'PaperlessBilling',
                                                    labelCheckedStyle={'color':'green'},
                                                    options=[{"label": v, "value": v } for v in df['PaperlessBilling'].unique()],
                                                    # labelClassName='m4',
                                                    value=[],
                                                    labelStyle={"display": "block"} ,
                                                )
                                                ,
                                                label=lab_var['PaperlessBilling'],
                                                # id="control-item-PaperlessBilling",
                                            )) ,
        
                                ]
                            ),               
                            dbc.Row(
                                [
                                dbc.Col(
                                    NamedGroup(
                                        dcc.RangeSlider(
                                            min=df['tenure'].min(),
                                            max=df['tenure'].max(),
                                            step=1,
                                            marks={i: '{}'.format(i) for i in range(0,73,12)},
                                            value=[0,72],
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            # updatemode='drag',
                                            id="tenure",
                                        ),
                                            
                                        label=lab_var['tenure'],
                                        # id="control-item-tenure",
                                        )
                                    ) 
                                ]
                            ),
                            dbc.Row(
                                [
                                dbc.Col(
                                    NamedGroup(
                                        dcc.RangeSlider(
                                            min=df['MonthlyCharges'].min(),
                                            max=df['MonthlyCharges'].max(),
                                            value=[df['MonthlyCharges'].min(),df['MonthlyCharges'].max()],
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            # updatemode='drag',
                                            id="MonthlyCharges",
                                        ),
                                            
                                        label=lab_var['MonthlyCharges'],
                                        # id="control-item-MonthlyCharges",
                                        )
                                    ) 
                                ]
                            ),
                            dbc.Row(
                                [
                                dbc.Col(
                                    NamedGroup(
                                        dcc.RangeSlider(
                                            min=df['TotalCharges'].min(),
                                            max=df['TotalCharges'].max(),
                                            value=[df['TotalCharges'].min(),df['TotalCharges'].max()],
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            # updatemode='drag',
                                            id="TotalCharges",
                                        ),
                                            
                                        label=lab_var['TotalCharges'],
                                        # id="control-item-TotalCharges",
                                        )
                                    ) 
                                ]
                            ),
                            
                            ]
                        )

                        ],
                        # className="sm",
                        # style={"text-align": "left","font-size": ".9vw"},
                    )


                    ],
                    className="header col-md-12 col-lg-3",
                    # id="cross-filter-options",
                    style={"text-align": "left","font-size": "70%",},
                ),
                
            ],
            className="row flex-display",
        ),
        
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(id="revenu-text"),
                                html.P(
                                    "Monthly Revenu",
                                    style={
                                        "text-align": "center",
                                        "color": "#909090",
                                        "font-weight": "bold",
                                    },
                                ),
                                html.P(
                                    id="revenu-number",
                                    style={
                                        "text-align": "center",
                                        # # "color": "#EC1E3D",
                                        "font-weight": "bold",
                                        "font-size": 25,
                                        },
                                ),
                                html.P(
                                    "Avg Monthly Charges",
                                    style={
                                        "text-align": "center",
                                        "color": "#909090",
                                        "font-weight": "bold",
                                    },
                                ),
                                html.P(
                                    id="avg-monthly-charge",
                                    style={
                                        "text-align": "center",   
                                        # "color": "purple",
                                        "font-weight": "bold",
                                        "font-size": 15,
                                    },
                                )
                            ],
                            className="card-header",
                            id="bg_id6",
                        ),

                        html.Div(
                            [
                                html.P(id="revenu-churn-text"),
                                html.P(
                                    "Revenu Churn",
                                    style={
                                        "text-align": "center",
                                        "color": "#909090",
                                        "font-weight": "bold",
                                    },
                                ),
                                html.P(
                                    id="revenu-churn-number",
                                    style={
                                        "text-align": "center",   
                                        "color": "purple",
                                        "font-weight": "bold",
                                        "font-size": 25
                                    },
                                ),
                                html.Br(),
                                html.P(
                                    "Revenu Churn rate",
                                    style={
                                        "text-align": "center",
                                        "color": "#909090",
                                        "font-weight": "bold",
                                    },
                                ),
                                html.P(
                                    id="revenu-churn-rate",
                                    style={
                                        "text-align": "center",   
                                        "color": "purple",
                                        "font-weight": "bold",
                                        "font-size": 15,
                                    },
                                )
                            ],
                            className="card-header",
                            id="bg_id7",
                        ),
                                

                    ],
                        id="right-column",
                        className="col-md-12 col-lg-3 mb-md-0 mb-4 card-chart-container",
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="rate-two-chart",figure={
                                "layout": {"title": "two rate Chart",
                                "height": 300, }})
                            ],
                            className="card-head",
                            id="bg_id8",
                        )
                    ],
                    # id="countGraphContainer",
                    className="col-md-12 col-lg-9 mb-md-0 mb-4 card-chart-container",
                            
                        ),
            ]
            ,className="row flex-display"
        ),
       
        # html.Div(
        #     [   html.Hr(),
        #         html.H5(
        #             "Sources",
        #             style={
        #                 "margin-top": "10",
        #                 "font-weight": "bold",
        #                 "text-align": "center",
        #             },
        #         ),
        #         dcc.Markdown(
        #             """\
        #                  - Design from: https://www.inetsoft.com/evaluate/bi_visualization_gallery/dashboard.jsp?dbIdx=1
        #                  - Data from IBM: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
        #                 """,
        #             style={"font-size": "10pt"},
        #         ),
        #     ],
        #     className="row ",
        # ),
    
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


#----------------------------------------- Background color -------------------------------
@callback(
   [Output("info-modal", "content_class_name"),Output("Variable", "style") ]+
   [Output("bg_id"+str(i), "style") for i in range(1,9)],
   Input("color_bg", 'data'),
)

def toggle_theme(data):
    if str(data)=="#fff":
        th = ['modalcontent1'] + [{'background-color': str(data)}]*9
    else:
        th = ['modalcontent2'] + [{'background-color': str(data)}]*9
    return th



# ---------------------------------------- Clear filter -------------------------------------------
@callback(Output("info-modal", "is_open"),
    Input("info-btn", "n_clicks"),
    State("info-modal", "is_open")
)
def toggle_modal(n_clicks, is_open):
    if (n_clicks):
        return not is_open
    return is_open
# -----------------------------------Data_frame filter update  -----------------------------------------------

@callback(
    [Output('data_frame','data'),Output('clear-button','disabled')]+[ Output(i,'options') for i in qual_var],
    [
        Input(i,'value') for i in df.columns[1:-3]
    ]
)
def filter_data(gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges, Churn):

   
  
    mask1 = df['gender'].isin(df['gender'].unique()) if len(gender)==0 else df['gender'].isin(gender)
    mask2 = df['SeniorCitizen'].isin(df['SeniorCitizen'].unique()) if len(SeniorCitizen)==0 else df['SeniorCitizen'].isin(SeniorCitizen)
    mask3 = df['Partner'].isin(df['Partner'].unique()) if len(Partner)==0 else df['Partner'].isin(Partner)
    mask4 = df['Dependents'].isin(df['Dependents'].unique()) if len(Dependents)==0 else df['Dependents'].isin(Dependents)
    mask5 = df['tenure'].between(tenure[0], tenure[1], inclusive='both')
    mask6 = df['PhoneService'].isin(df['PhoneService'].unique()) if len(PhoneService)==0 else df['PhoneService'].isin(PhoneService)
    mask7 = df['MultipleLines'].isin(df['MultipleLines'].unique()) if len(MultipleLines)==0 else df['MultipleLines'].isin(MultipleLines)
    mask8 = df['InternetService'].isin(df['InternetService'].unique()) if len(InternetService)==0 else df['InternetService'].isin(InternetService)
    mask9 = df['OnlineSecurity'].isin(df['OnlineSecurity'].unique()) if len(OnlineSecurity)==0 else df['OnlineSecurity'].isin(OnlineSecurity)
    mask10 = df['OnlineBackup'].isin(df['OnlineBackup'].unique()) if len(OnlineBackup)==0 else df['OnlineBackup'].isin(OnlineBackup)
    mask11 = df['DeviceProtection'].isin(df['DeviceProtection'].unique()) if len(DeviceProtection)==0 else df['DeviceProtection'].isin(DeviceProtection)
    mask12 = df['TechSupport'].isin(df['TechSupport'].unique()) if len(TechSupport)==0 else df['TechSupport'].isin(TechSupport)
    mask13 = df['StreamingTV'].isin(df['StreamingTV'].unique()) if len(StreamingTV)==0 else df['StreamingTV'].isin(StreamingTV)
    mask14 = df['StreamingMovies'].isin(df['StreamingMovies'].unique()) if len(StreamingMovies)==0 else df['StreamingMovies'].isin(StreamingMovies)
    mask15 = df['Contract'].isin(df['Contract'].unique()) if len(Contract)==0 else df['Contract'].isin(Contract)
    mask16 = df['PaperlessBilling'].isin(df['PaperlessBilling'].unique()) if len(PaperlessBilling)==0 else df['PaperlessBilling'].isin(PaperlessBilling)
    mask17 = df['PaymentMethod'].isin(df['PaymentMethod'].unique()) if len(PaymentMethod)==0 else df['PaymentMethod'].isin(PaymentMethod)
    mask18 = df['MonthlyCharges'].between(MonthlyCharges[0], MonthlyCharges[1], inclusive='both')
    mask19 = df['TotalCharges'].between(TotalCharges[0], TotalCharges[1], inclusive='both')
    mask20 = df['Churn'].isin(df['Churn'].unique()) if len(Churn)==0 else df['Churn'].isin(Churn)

    mask = (mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & mask8 & mask9 & mask10 & mask11 
                & mask12 & mask13 & mask14 & mask15 & mask16 & mask17 & mask18 & mask19 & mask20)

    df_filter = df[mask]



    dis = df_filter.shape == df.shape

    # if not dis:n_clicks=0

    if list(InternetService)==['No'] :
        checklist_update = [
            [ {"label": v, "value": v ,'disabled':True if v=='No P-S' else i in Services+['PhoneService'] } for v in df[i].unique()] for i in qual_var  
        ]
        

    elif 0<len(OnlineBackup)<3:
        Other_services = Services.copy()
        Other_services.remove('OnlineBackup')
        checklist_update = [
                    [ {"label": v, "value": v ,'disabled': False  if i not in Other_services+ ['PhoneService','MultipleLines','InternetService'] \
                else  (True if (v== 'No I-S' ) else v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var
        ]


    elif 0<len(DeviceProtection)<3:
        Other_services = Services.copy()
        Other_services.remove('DeviceProtection')
        checklist_update = [
                    [ {"label": v, "value": v ,'disabled': False  if i not in Other_services+ ['PhoneService','MultipleLines','InternetService'] \
                else  (True if (v== 'No I-S' ) else v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var
        ]


    elif 0<len(TechSupport)<3:
        Other_services = Services.copy()
        Other_services.remove('TechSupport')
        checklist_update = [
                    [ {"label": v, "value": v ,'disabled': False  if i not in Other_services+ ['PhoneService','MultipleLines','InternetService'] \
                else  (True if (v== 'No I-S' ) else v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var
        ]

    elif 0<len(StreamingTV)<3:
        Other_services = Services.copy()
        Other_services.remove('StreamingTV')
        checklist_update = [
                    [ {"label": v, "value": v ,'disabled': False  if i not in Other_services+ ['PhoneService','MultipleLines','InternetService'] \
                else  (True if (v== 'No I-S' ) else v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var

        ]

    elif 0<len(StreamingMovies)<3:
        Other_services = Services.copy()
        Other_services.remove('StreamingMovies')
        checklist_update = [
                    [ {"label": v, "value": v ,'disabled': False  if i not in Other_services+ ['PhoneService','MultipleLines','InternetService'] \
                else  (True if (v== 'No I-S' ) else v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var
        ]

    elif 0<len(OnlineSecurity)<3:
        Other_services = Services.copy()
        Other_services.remove('OnlineSecurity')
        checklist_update = [
                    [ {"label": v, "value": v ,'disabled': False  if i not in Other_services+ ['PhoneService','MultipleLines','InternetService'] \
                else  (True if (v== 'No I-S' ) else v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var
        ]


    elif len(PhoneService)==1:
        checklist_update = [
            [ {"label": v, "value": v ,'disabled': False  if i not in Services+ ['PhoneService', 'MultipleLines','InternetService'] \
                else ( True if v== 'No P-S' else v not in df_filter[i].unique() )  } for v in df[i].unique()] for i in qual_var
        ]

    elif len(MultipleLines)==1:
        checklist_update = [
            [ {"label": v, "value": v ,'disabled': False  if i not in Services+ ['PhoneService','InternetService'] \
                else  (v not in df_filter[i].unique() ) } for v in df[i].unique()] for i in qual_var
        ]


    else:
        checklist_update= [
            [ {"label": v, "value": v ,'disabled': (v not in df_filter[i].unique()) and (df[i].nunique()-df_filter[i].nunique()==1) } for v in df[i].unique()] for i in qual_var
        ]
        
    liste = [df_filter.to_json(date_format='iso',orient='split'), dis] +checklist_update
    return liste


#------------------------------------ clear-filter----------------------------------------------

@callback(
    [Output(i,'value') for i in df.columns[1:-3]],
    [Input('clear-button','n_clicks')],
)

def Clear_filter(n_clicks):
    checklist_value_update= [
        [],[],[],[],[0,72],[],[],[],[],[],[],[],[],[],[],[],[],
        [df['MonthlyCharges'].min(),df['MonthlyCharges'].max()],[df['TotalCharges'].min(),df['TotalCharges'].max()],[]
    ]
    return  checklist_value_update

# ----------------------------------- Contract_chart I ---------------------------------------
@callback(
    Output("c-chart", "figure"),
    [
        Input("Variable", "value"),
        Input("data_frame", "data"),
    ]
)
def contract_update(Variable,dff):
    df = pd.read_json(dff,orient='split')
    if Variable==None:Variable='Contract'
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True
    )

    fig.add_trace(go.Bar(x=df.groupby(Variable)['MonthlyCharges'].sum(),
                         y=df.groupby(Variable)['MonthlyCharges'].sum().index,
                         orientation='h',#text=['',"","u"], textposition=  "bottom center" ,
                        ),
                  row=1, col=1
                 )

    fig.add_trace(go.Bar(x=df.groupby(Variable)['MonthlyCharges'].mean(),
                         y=df.groupby(Variable)['MonthlyCharges'].mean().index,
                         orientation='h',
                        ),
                  row=1, col=2
                 )

    fig.add_trace(go.Bar(x=df.groupby(Variable)['tenure'].mean(),
                             y=df.groupby(Variable)['tenure'].mean().index,
                             orientation='h',
                            ),
                      row=1, col=3
                     )


    fig.update_traces(marker_color=['lightsalmon','lightseagreen','royalblue','lightpink'])



    annotations = []
    annotations.append(dict(xref='paper', yref='paper',
                            x=.03, y=1.2,
                            text='Total Revenu',
                                 font=dict(size=14,color="#909090"),
                            showarrow=False))
    annotations.append(dict(xref='paper', yref='paper',
                            x=.5, y=1.2,
                            text='AVG Revenu',
                                 font=dict(size=14,color="#909090"),
                            showarrow=False))
    annotations.append(dict(xref='paper', yref='paper',
                            x=.97, y=1.2,
                            text='AVG Tenure',
                                 font=dict(size=14,color="#909090"),
                            showarrow=False))
    fig.update_layout(annotations=annotations)

    fig.update_layout(
            title=f"<b> Customers Profile by {lab_var[Variable]} <b>",
            # template="simple_white",
            bargroupgap=0.3, 
            font={'size': 12, 'family':'garamond'},
            showlegend=False,
            plot_bgcolor="rgb(0,0,0,0)",
            legend_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
            font_color="#909090",
                )       

    fig.update_xaxes(axis_font)
    

    return fig

# ----------------------------------- Service_chart I ---------------------------------------
@callback(
    Output("service-chart", "figure"),
    [
        Input("Variable", "value"),
        Input("data_frame", "data"),
    ]
)
def ServiceUpdate(Variable,dff):
    df = pd.read_json(dff,orient='split')
    if Variable==None:Variable='Contract'
    
    data = (df.groupby(['InternetService',Variable],as_index=False)
            .agg({"MonthlyCharges":'mean','customerID':'count'})
            .rename(columns={'customerID':'Frequency'})
           )

    data1 = (df.groupby(['PhoneLine',Variable],as_index=False)
            .agg({"MonthlyCharges":'mean','customerID':'count'})
            .rename(columns={'customerID':'Frequency'})
           )
    Li=[]
    for i in Services:
            df1 = df.groupby([Variable,i])["Churn"].agg({'count'}).reset_index()
            df1[i] = df1[i].replace({u:'{}: {}'.format(lab_var[i],u) for u in df[i].value_counts().index})
            df1.rename(columns={i:'variable'},inplace=True)
            Li+=[df1[df1.apply(lambda x: not (('No' in x['variable']) or ('0' in x['variable']) ) ,axis=1)]]
    d= pd.concat(Li)
    d.replace({'Device Protection: Yes':'Protect','Online Backup: Yes':'Backup','Tech Support: Yes':'Tech',
          'Streaming T V: Yes':'TV','Online Security: Yes':'Security','Streaming Movies: Yes':'Movies'},inplace=True)
    
    fig = make_subplots(
        rows=1, cols=3,specs=[[{}, {'type': 'polar'}, {}]]
        # shared_yaxes=True
    )    
    for s ,t in enumerate(data[Variable].unique()):
        df1=data[data[Variable]==t]
        fig.add_trace(
            go.Bar(
                x=df1['InternetService'].astype(str),
                y=df1["Frequency"],
                name=t,
                marker=dict(color=['lightsalmon','lightseagreen','royalblue','lightpink'][s]),
            ),
            
            row=1 , col= 1,
        )

        d1=d[d[Variable]==t]
        # print(rlist)
        fig.add_trace(
            go.Scatterpolar(
                name = t,
                r = d1['count'].values.tolist()+d1['count'].values.tolist()[:1],
                theta = d1['variable'].values.tolist()+d1['variable'].values.tolist()[:1],
                marker=dict(color=['lightsalmon','lightseagreen','royalblue','lightpink'][s]),
            ),
                      
            row=1 , col= 2,
        )

        df2=data1[data1[Variable]==t]
        fig.add_trace(
            go.Bar(
                x=df2['PhoneLine'].astype(str),
                y=df2["Frequency"],
                name=t,
                marker=dict(color=['lightsalmon','lightseagreen','royalblue','lightpink'][s]),
            ),
            
            row=1 , col= 3,
        )
        
    fig.update_polars(bgcolor='aliceblue',)

    annotations = []
    annotations.append(dict(xref='paper', yref='paper',
                            x=0.12, y=1.2,
                            text='Internet',
                                 font=dict(size=14,color="#909090"),
                            showarrow=False))
    annotations.append(dict(xref='paper', yref='paper',
                            x=.5, y=1.2,
                            text='Other Service',
                                 font=dict(size=14,color="#909090"),
                            showarrow=False))
    annotations.append(dict(xref='paper', yref='paper',
                            x=.9, y=1.2,
                            text='Phone Line',
                                 font=dict(size=14,color="#909090"),
                            showarrow=False))
    fig.update_layout(annotations=annotations)

    fig.update_layout(
            title=f'<b> Service Purchase by {lab_var[Variable]} <b>',
            polar_radialaxis_ticks= "",
            polar_radialaxis_showticklabels= False,
            font={'size': 12, 'family':'garamond'},
            bargroupgap=0.5, 
            barmode='stack',
            # template='simple_white',
            showlegend=False,
            plot_bgcolor="rgb(0,0,0,0)",
            legend_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgb(0,0,0,0)",
            font_color="#909090",
             )
    fig.update_yaxes(showticklabels=False,visible=False)
    # fig.update_xaxes(showticklabels=False,visible=False)
    return fig

# ----------------------------------- Rate_chart I ---------------------------------------
@callback(
    Output("rate-chart", "figure"),
    [
        Input("Variable", "value"),
        Input("data_frame", "data"),
    ]
)
def RateUpdate(Variable,dff):
    df = pd.read_json(dff,orient='split')
    if Variable==None:Variable='Contract'
    data = df.groupby([Variable,'tenure'])['churn1'].mean().mul(100).reset_index()
    data1 = df.groupby(['tenure'])['churn1'].mean().mul(100).reset_index()

    fig = px.scatter(data, x="tenure", y="churn1",
                    color=Variable, 
                    facet_col=Variable,
                    labels={'churn1':'Churn rate'},
                    color_discrete_map={i :x for i ,x in zip( data[Variable].unique(),
                     ['lightsalmon','lightseagreen','royalblue','lightpink'])})
    fig.add_trace(
        
        go.Scatter(
            x=data1['tenure'],
            y=data1["churn1"],
            mode='markers',
            line_color='red',
            visible=False,
    ),
        row="all", col="all",
    )

    fig.update_layout(
        title=f'<b> Churn rate trend by {lab_var[Variable]} Type <b>',
        title_x=0.7,
        yaxis_ticksuffix ='%',
        font={'size': 12, 'family':'garamond'},
        showlegend=False,
        template='simple_white',                    
        plot_bgcolor="rgb(0,0,0,0)",
        legend_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgb(0,0,0,0)",
        font_color="#909090",
    )
    fig.update_xaxes(axis_font)
    # fig.update_yaxes(axis_font)
        
    fig.update_yaxes(matches='y',linecolor='rgb(204, 204, 204)',)

    # # buttons for updatemenu
    buttons = [dict(method='restyle',
                    label='Overall trend',
                    visible=True,
                    args=[{'label': 'overal',
                        'visible':True ,
                        }
                        ]),
            dict(method='restyle',
                    label='No',
                    visible=True,
                    args=[{'label': 'no',
                        'visible':[True]* df[Variable].nunique()+[False]* df[Variable].nunique(),
                        }
                        ])]
    # subplot titles
    for anno in fig['layout']['annotations']:
        anno['text']=''
    # specify updatemenu        
    um = [{'buttons':buttons,
        'type':'buttons',
        # 'size':'7',
        'showactive': True,
        'active': 1,
        'direction': 'right',
        'x': .1,'y': 1.4,}
        ]

    fig.update_layout(updatemenus=um)
    



    return fig

# ----------------------------------- Rate_chart II ---------------------------------------
@callback(
    Output("rate-two-chart", "figure"),
    [
        Input("Variable", "value"),
        Input("data_frame", "data"),
    ]
)
def Rate2Update(Variable,dff):
    df = pd.read_json(dff,orient='split')

    data = (df[df['Churn']=='Yes'].groupby(['tenure'])['MonthlyCharges'].sum()/df.groupby(['tenure'])['MonthlyCharges'].sum()).mul(100)
    data1 = df.groupby(['tenure'])['churn1'].mean().rename('rate').replace({0:np.nan}).mul(100)
    data= pd.concat([data,data1],axis=1).reset_index()

    
    # Creating two subplots
    fig = make_subplots(rows=2, cols=1, specs=[[{}],
                                            [{}]], 
                        shared_xaxes=True,
                        y_title='<br> Rate <br>',
        
                    )
                
    fig.append_trace(go.Bar(
        y=data['rate'] ,
        x=data['tenure'],
        marker=dict(
            color='red',
            line=dict(
                color='rgba(300, 0, 0, 1.0)',
                # width=1
            ),
        ),
        name='Churn rate',
    ), 2, 1)

    fig.append_trace(go.Bar(
        y=data['MonthlyCharges'] , x=data['tenure'],
        marker=dict(
            color='purple',
            line=dict(
                color='rgba(100, 0, 100, 1.0)',
                # width=1
            ),
        ),
        name='Revenu Churn rate',
    ), 1, 1)

    fig.update_layout(
        title='Churn rate & Revenu Churn rate',
        legend=dict(x=0.8, y=1.5, font_size=10),
        font={'size': 12, 'family':'garamond'},
        template='simple_white',
        plot_bgcolor="rgb(0,0,0,0)",
        legend_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgb(0,0,0,0)",
        font_color="#909090",
    )

    fig.update_xaxes(title_text="<b> Tenure </b>", row=2,linecolor='rgb(204, 204, 204)',)
    fig.update_yaxes(ticksuffix  = '%', row=2,linecolor='rgb(204, 204, 204)',)
    fig.update_yaxes(ticksuffix  = '%', row=1,linecolor='rgb(204, 204, 204)',)



    return fig
# ----------------------------------- Cases_text ---------------------------------------

@callback(
    [
        Output("case-number", "children"),
        Output("churn-number", "children"),
        Output("churn-rate", "children"),
    ],
    [
        Input("Variable", "value"),
        Input("data_frame", "data"),
    ],
)

def update_card1(Variable,dff):
    dff = pd.read_json(dff,orient='split')
    case  = dff.shape[0]
    churn = sum(dff['Churn']=='Yes')
    rate  = '{:,.0f} %'.format(churn/case*100) if case!=0 else np.nan()
    return case,churn,rate

# ----------------------------------- Revenu_text ---------------------------------------

@callback(
    [
        Output("revenu-number", "children"),
        Output("avg-monthly-charge", "children"),
        Output("revenu-churn-number", "children"),
        Output("revenu-churn-rate", "children"),
    ],
    [
        Input("Variable", "value"),
        Input("data_frame", "data"),
    ],
)

def update_card2(Variable,dff):
    dff = pd.read_json(dff,orient='split')
    revenu  =  sum(dff['MonthlyCharges'])/1000
    avg_revenu= np.mean(dff['MonthlyCharges'])
    revenu_churn =   sum(dff[dff['Churn']=='Yes']['MonthlyCharges'])/1000
    revenu_rate  = '{:,.0f} %'.format(revenu_churn/revenu*100) if revenu!=0 else np.nan() 
    return '{:,.2f}k $' .format(revenu),'{:,.2f} $' .format(avg_revenu),'{:,.2f}k $' .format(revenu_churn),revenu_rate

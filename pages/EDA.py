from __future__ import annotations

import dash
from dash_bootstrap_templates import load_figure_template


# dash.register_page(__name__, path='/', name='EDA')
dash.register_page(
    __name__,
    path='/EDA',
    title="Customer Churn",
)

load_figure_template("minty")

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
        # ======================= Attributes Selection
        dbc.Row(
            [
                dbc.Col([

                    html.H2(
                        "Telecom Customer Churn : Univariate and Multivariate Analysis",
                        style={"font-weight": "bold","color": "#0084d6"},
                        className="text-center mb-4",
                    ),

                    html.H6(
                        "Analysis of the relationship between Monthly Charges and \n the Churn status within the Customers Caracteristiques",
                        className="text-center",#text-info mb-4
                        style={"margin-buttom": "5px"},
                    ),
                    # width=6,
                ]
                ),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        # dbc.Container(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            dbc.Label("Attributes", className="text-center text-primary mb-4"),
                                        ),
                                        html.Div(
                                            [dcc.Dropdown(
                                                id="Attribute",
                                                searchable=False,
                                                clearable=False,    
                                                multi=False,
                                                placeholder="Select Attribute",
                                                options=[
                                                    {"label": lab_var[c], "value": c}
                                                    for c in df.columns[1:-4] 
                                                ],
                                                value="gender",
                                            )],
                                            style={"color":"#0a66c2"},

                                        ),
                           
                                    ],
                                    className="text-center col-md-12 col-lg-3 mb-md-0 mb-4 card-chart-contain",
                                    # style={"margin": 30,'display':'center'},


                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Label("Lifespan with the company (in Months)",className="text-center text-primary mb-4"),
                                            dcc.RangeSlider(min=df['tenure'].min(),
                                                max=df['tenure'].max(),
                                                step=1,
                                                marks={i: '{}'.format(i) for i in range(0,73,12)},
                                                value=[0,72],
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                # updatemode='drag',
                                                id="my-rangeslider"),

                                            # dcc.Graph(id='my-graph')
                                        ],
                                        className="text-center col-md-12 col-lg-9 mb-md-0 mb-4 card-chart-contain",
                                    ),
                                ),
                            ],
                            className="row flex-display-center",
                            style={"margin-top": 30,"margin-left": 200,}


                    ), 
                ),
            ],
        ),
        
         # ======================  uni & bi Chart
        html.Div(
            [
                dbc.Col(
                    [dbc.Card(
                        dbc.CardBody(html.Div(id="uni_chart")), className="card-header",id="bg_id9",
                    ),
                   ],
                   className="col-md-12 col-lg-5  card-chart-contain",

                ),
                dbc.Col(
                    [dbc.Card(
                        dbc.CardBody(html.Div(id="bi_chart")), className="card-header",id="bg_id10",
                    ),
                    ],
                    className="col-md-12 col-lg-7  card-chart-contain",
                ),
            ],
            className="row flex-display",

        ), 
        # ====================== tri Chart 
        dbc.Row(
            [     
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(html.Div(id="tri_chart")), className="card-header",id="bg_id11",
                    ),
                ),
            ], className="p-3"
        ),                           
       
       
    ]
)

@callback(
    [Output("Attribute", "style")]+
   [Output("bg_id"+str(i), "style") for i in range(9,12)],
   Input("color_bg", 'data'),
)

def toggle_theme(data):
    th = [{'background-color': str(data)}]*4
    return th
# ----------------------------------- Graph I ---------------------------------------
@callback(
    Output(component_id="uni_chart", component_property="children"),
    Input(component_id="Attribute", component_property="value"),
    Input(component_id="my-rangeslider", component_property="value"),
    )


def Univariate(Attribute,myrange):
    Attribute = 'gender' if Attribute == None else Attribute
    df1=df[df['tenure'].between(myrange[0], myrange[1], inclusive='both')]
    X=df1[Attribute]
    title =lab_var[Attribute]
    cond = X.dtype == 'O' or X.nunique()<5
    if cond:
        data=X.value_counts(normalize=True)
    
        fig = go.Figure(data=[go.Pie(labels=data.index, 
                                values=data, 
                                # textinfo = 'label+percent',
                                textposition = 'inside',
                                insidetextfont={'color':'white'},
                                # pull=[0.1, 0, 0.2,0, 0,0]
                                )])
        # fig.update_traces(
        #     # textfont_size=10,
        #     marker=dict(
        #             colors=['royalblue','gray','yellow','green'],
        #         ),
        #     )
    else:
        fig = px.box(df1, x=Attribute,)
        

    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        xaxis=axis_font,
        xaxis_title= f"<b>  {title}  <b>",
        title_text= f"<b>{title}<b>" if cond else f"<b>  Customer's {title} Distribution <b>",
        titlefont={'size': 12, 'family': 'garamond'},
        title_x=0.5,
        plot_bgcolor="rgb(0,0,0,0)",
        legend_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgb(0,0,0,0)",
        font_color="#909090",
        hovermode="closest",
        # width=500,
        height=300,
    )
    return html.Div(dcc.Graph(figure=fig), id="uni_chart")

# ----------------------------------- Graph II ---------------------------------------
@callback(
    Output(component_id="bi_chart", component_property="children"),
    Input(component_id="Attribute", component_property="value"),
    Input(component_id="my-rangeslider", component_property="value"),
    )
def Bivariate(Attribute,myrange):
    Attribute = 'gender' if Attribute == None else Attribute
    df1=df[df['tenure'].between(myrange[0], myrange[1], inclusive='both')]
    title = lab_var[Attribute]
    
    if Attribute=="Churn":return None
    elif Attribute not in ['MonthlyCharges','TotalCharges','tenure']:
        data = (df1.groupby([Attribute,'Churn'],as_index=False)
                .agg({"MonthlyCharges":'mean','customerID':'count'})
                .rename(columns={'customerID':'Frequency'})
                # .sort_values('Churn')
                # .reset_index(drop=True)
            )
        fig =  px.bar(data,x=data[Attribute].astype(str),y='Frequency',color='Churn',barmode='group',text_auto='text',        
                color_discrete_map={ 
                    "No": "RebeccaPurple", "Yes": "lightsalmon"
                },
                 )

        fig.update_layout(
                # title=f'<b> Customer numbers & Average Monthly Charges by {title} <b>',
                xaxis_title=f'<b>{title}<b>',
                yaxis_title='<b>Number of Customers <b>',
                    legend_title='<b>Churn<b>',
                # titlefont={'size': 15, 'family':'garamond'},
                font={'size': 12, 'family':'garamond','color':'black'},
            
                    title_x=0.5,
                    template='simple_white',
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgb(0,0,0,0)",
                    font_color="#909090",
                    # hovermode="closest",
                # width=500,
                 height=300,
                )
        # fig.update_yaxes(title_text="<b> Average Monthly Charges</b>", secondary_y=True)
   
    else:
        fig  = px.box(df1, x=Attribute,color='Churn',
            color_discrete_map={ 
                    "No": "RebeccaPurple", "Yes": "lightsalmon"
                },
                )

        fig.update_layout(
        xaxis=axis_font,
        xaxis_title=lab_var[Attribute],
        # yaxis=axis_font,
        # title_text=f"<b>  Customer’s {title} Distribution <b>",
        titlefont={'color':'black', 'size': 16, 'family': 'San-Serif'},
         title_x=0.5,
        # template='simple_white',
        plot_bgcolor="rgb(0,0,0,0)",
        legend_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgb(0,0,0,0)",
        font_color="#909090",
        # hovermode="closest",
        # width=500,
        height=300
        )
    return html.Div(dcc.Graph(figure=fig), id="bi_chart")

# # ----------------------------------- Graph III ---------------------------------------
@callback(
    Output(component_id="tri_chart", component_property="children"),
    Input(component_id="Attribute", component_property="value"),
    Input(component_id="my-rangeslider", component_property="value"),
    )
def Trivariate(Attribute,myrange):

    Attribute = 'gender' if Attribute == None else Attribute
    df1=df[df['tenure'].between(myrange[0], myrange[1], inclusive='both')]

# data to plot:   distribution churn vs not-churn
    if Attribute  in ['MonthlyCharges','TotalCharges','tenure']:
        surv = df1[df1['Churn'] == 'Yes'][Attribute]
        vict = df1[df1['Churn'] == "No"][Attribute]


        group_labels = ['Yes', 'No']
        fig = ff.create_distplot([surv, vict],
                                group_labels, 
                                show_hist=False, 
                                show_rug=False,
                                colors=["lightsalmon","RebeccaPurple"]
            
                                )
        u=df1[Attribute].describe().values[1:]
        for i,y in enumerate (u):
            if i==0:
                # print(i)
                fig.add_vline(x=y,line_width=3,line_dash='dash',line_color='green',
                            annotation_text='Mean',annotation_position="bottom",)
            elif i>=2 and i<6:
                # print(['','', 'q1','q2','q3','q4','q5'][x],y)

                fig.add_vrect(
                    x0=y, x1=u[i+1],
                    annotation_text= ["Q1","Q2","Q3" , "Q4"][i-2],
                    annotation_position="top",
                    fillcolor=["skyblue","Salmon","lightgray","gray"][i-2],
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                ),
        p='months'if Attribute=='tenure' else '$'

        fig.update_layout(
                        height=400,
                        title='<b>Customers {} distn. by Churn status<b>'.format(lab_var[Attribute]),
                        xaxis_title= lab_var[Attribute]+f'[{p}]', 
                        yaxis_title='Density',
                        titlefont={'size': 24},
                        font_family = 'San Serif',
                        # width=900,height=500,
                        template="simple_white",
                        legend_title='Churn',
                        showlegend=True,
                        plot_bgcolor="rgb(0,0,0,0)",
                        legend_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgb(0,0,0,0)",
                        font_color="#909090",
                        # font=dict(
                        #     color ='black',
                        #     ),
                        legend=dict(
                            orientation="v",
                            y=1, 
                            yanchor="bottom", 
                            x=1.0, 
                            xanchor="right",)   
        )
    else:
        fig = px.box(df1,x=Attribute, y='MonthlyCharges',color='Churn',
        color_discrete_map={ 
                    "No": "RebeccaPurple", "Yes": "lightsalmon"
                },)
        fig.add_hline(y=df1['MonthlyCharges'].mean(),line_width=3,line_dash='dash',line_color='green',
                      annotation_text='Mean',annotation_position="left",layer="below",)

        data = (df1.groupby([Attribute,'Churn'],as_index=False)
                .agg({"MonthlyCharges":'mean','customerID':'count'})
                .rename(columns={'customerID':'Frequency'}))

        for t in data['Churn'].unique():
            df2=data[data['Churn']==t]
            fig.add_trace(
                go.Scatter(
                    x=df2[Attribute],
                    y=df2["MonthlyCharges"],
                    showlegend=False,
                    # name=t,
                    marker=dict(color=["lightsalmon","RebeccaPurple"][t=='No']),
                    # marker=dict(size=10, line=dict(width=2, color=["orange", 'orangered'][t=='Yes']) ),
                    # line=dict(width=2, color="rgb(102, 255, 204)"),
                    text=df2["MonthlyCharges"],
                    textposition=  "middle left" if t=='No' else "middle right",
                    # textfont=dict(color="darkblue"),
                    mode="markers+text",
                    texttemplate="%{text:.2s}$",
                ),)

        fig.update_layout(
                        title='<b> {} & Average Monthly Charges by Churn status <b>'.format(lab_var[Attribute]),
                        xaxis_title= lab_var[Attribute], 
                        yaxis_title=lab_var['MonthlyCharges'],
                        titlefont={'size': 24},
                        font_family = 'San Serif',
                        # width=900,height=500,
                        template="simple_white",
                        legend_title='Churn',
                        showlegend=True,
                        plot_bgcolor="rgb(0,0,0,0)",
                        legend_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgb(0,0,0,0)",
                        font_color="#909090",
                        height=400,

 
        )

    return html.Div(dcc.Graph(figure=fig), id="tri_chart")

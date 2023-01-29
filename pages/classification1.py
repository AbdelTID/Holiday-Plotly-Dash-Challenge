# Import required libraries
import re
import pathlib
# import dash_table
from dash import dash_table
import dash
import math
import pandas as pd
import numpy as np
import datetime as dt
import base64
import io
import base64
import seaborn as sns
import matplotlib as plt
import pandas as pd
import dash_daq as daq
import plotly.express as px
import plotly.figure_factory as ff
from dash import  dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import Dash, Input, Output, callback, dcc, html,State
import dash_loading_spinners as dls



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

models = ['SVC', 'Random Forest', 'XGB', 'DT', 'Logistic']
models_dict = {
    "Logistic":LogisticRegression(),'Random Forest':RandomForestClassifier(random_state=42,max_depth=6),"XGB":XGBClassifier(random_state=42,max_depth=6),
    "DT":DecisionTreeClassifier(random_state=42,max_depth=6),"SVC":SVC(probability=True)
}
FONTSIZE = 15
FONTCOLOR = "#F5FFFA"
BGCOLOR ="#3445DB"

dash.register_page(
    __name__,
    path='/classification1',
    title="Customer Churn",
)

# app = dash.Dash(
#     __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
# )


# server = app.server
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

reg_df = df[df.columns[1:-3]].copy()

binary_feat = reg_df.nunique()[reg_df.nunique() == 2].keys().tolist()
numeric_feat = [col for col in reg_df.select_dtypes(['float','int']).columns.tolist() if col not in binary_feat]
categorical_feat = [ col for col in reg_df.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]

df_proc = reg_df.copy()

features_obj = reg_df.columns


le = LabelEncoder()
for i in binary_feat:
  df_proc[i] = le.fit_transform(df_proc[i])

df_proc = pd.get_dummies(df_proc, columns=categorical_feat)


PAGE_SIZE = 10
# Create app layout
layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
    
               html.H2(
                                    "Telecom Customer Churn : Classification Models",
                                    style={"font-weight": "bold","color": "#0084d6",},
                                    className="text-center mb-4",

                                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        
        html.Hr(),

        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Train / Test  Split",
                            # className="control_label",
                        ),
        #                 upload_layout,
        #                 html.Div(id='slider-output-container'),
                        html.Br(),
                        html.Br(),

   
                        daq.Slider(
                            id = 'slider',
                            min=50,
                            max=90,
                            value=80,
                            handleLabel={"showCurrentValue": True,"label": "SPLIT"},
                            step=1
                        ),
                        html.Br(),
                        # html.Br(),
                        html.P("Sampling",className="control_sample"),
                        dcc.RadioItems(
                            id= "select_sample",
                            options=["Initial","Under","Over"],
                            value="Initial",
                            # inline=True
                            className="dcc_control",
                            inputStyle={"margin-left":20,"margin-right":3,}
                        ),
                        html.Br(),

                       
                        html.Br(),
                        html.P("Models", className="control_label"),
                        dcc.Dropdown(
                            id="select_models",
                            options = [{'label':x, 'value':x} for x in models],
                            value = 'Logistic',
                            # multi=True,
                            clearable=False,
                            className="text-primary",

                        ),
                        # html.Div(
                        #     id = 'best-model', style={'color': 'blue', 'fontSize': 15} 
                        # ),
                        html.Br(),
                        
                        html.Br(),
                        

                                                                                               
                    ],
                    className="card-chart-contain col-md-12 col-lg-3",
                    # id="cross-filter-options",
                ),
        #         #--------------------------------------------------------------------------------------------------------------------
                html.Div(
                    [
                         html.Div(
                            daq.LEDDisplay(
                                id='records',
                                #label="Default",
                                value=0,
                                label = "Records",
                                size=FONTSIZE,
                                color = FONTCOLOR,
                                backgroundColor=BGCOLOR
                            )
                        ),
                        dbc.Tooltip("Number of Observation in\n the Overall dataset", target="records",placement='bottom'),
                        html.Br(),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div([daq.LEDDisplay(
                                            id='trainset',
                                            #label="Default",
                                            value=0,
                                            label = "Train",
                                            size=FONTSIZE,
                                            color = FONTCOLOR,
                                            backgroundColor=BGCOLOR,
                                            
                                        )],style={'display':'inline-block',"margin-right":10} ),
                                        html.Div([daq.LEDDisplay(
                                            id='testset',
                                            #label="Default",
                                            value=0,
                                            label = "Test",
                                            size=FONTSIZE,
                                            color = FONTCOLOR,
                                            backgroundColor=BGCOLOR, )],style={'display':'inline-block'}
                                        ),
                                    ],style={'display':'flex-center'}, 
                                    className="text-center mb-4",

                                    # className="row flex-display",
                                )
                            ]
                        ),
                        # html.Br(),
                        dbc.Tooltip("Number of Observation in\n the Train dataset", target="trainset",placement='bottom'),
                        dbc.Tooltip("Number of Observation in\n the Test dataset", target="testset",placement='bottom'),

                        html.P("Accuracy:",className="text-center"),
                        html.Div([
                            html.Div(
                                [
                                    html.Div([daq.LEDDisplay(
                                        id='acc_baseline',
                                        #label="Default",
                                        value=0,
                                        label = "Baseline",
                                        size=FONTSIZE,
                                        color = FONTCOLOR,
                                        backgroundColor=BGCOLOR
                                    )],style={'display':'inline-block',"margin-right":10}
                                    ),
                                    html.Div([daq.LEDDisplay(
                                        id='acc_train',
                                        #label="Default",
                                        value=0,
                                        label = "Training",
                                        size=FONTSIZE,
                                        color = FONTCOLOR,
                                        backgroundColor=BGCOLOR
                                    )],style={'display':'inline-block',"margin-right":10}
                                    ),
                                    html.Div([daq.LEDDisplay(
                                        id='acc_test',
                                        #label="Default",
                                        value=0,
                                        label = "Test",
                                        size=FONTSIZE,
                                        color = FONTCOLOR,
                                        backgroundColor=BGCOLOR
                                    )],style={'display':'inline-block'})
                                ],style={'display':'flex-center'}, className="text-center mb-4",

                            )
                        ]

                        ),

                        dbc.Tooltip("Accuracy to beat, \n get from the original dataset", target="acc_baseline",placement='bottom'),
                        dbc.Tooltip("Training Accuracy", target="acc_train",placement='bottom'),
                        dbc.Tooltip("Test Accuracy", target="acc_test",placement='bottom'),

                        # html.Br(),
                        html.Div(
                            [
                                html.Div(
                                  [
                                        html.Div([daq.LEDDisplay(
                                            id='precision',
                                            #label="Default",
                                            value=0,
                                            label = "Precision",
                                            size=FONTSIZE,
                                            color = FONTCOLOR,
                                            backgroundColor=BGCOLOR
                                        )],style={'display':'inline-block',"margin-right":10}
                                        ),
                                        html.Div([daq.LEDDisplay(
                                                id='recall',
                                                #label="Default",
                                                value=0,
                                                label = "Recall",
                                                size=FONTSIZE,
                                                color = FONTCOLOR,
                                                backgroundColor=BGCOLOR
                                        )],style={'display':'inline-block',"margin-right":10}),
                                        html.Div([daq.LEDDisplay(
                                            id='auc',
                                            #label="Default",
                                            value= 0,
                                            label = "AUC",
                                            size=FONTSIZE,
                                            color = FONTCOLOR,
                                            backgroundColor=BGCOLOR
                                        )],style={'display':'inline-block'}),  
                                         
                                    ],style={'display':'flex-center'}, className="text-center mb-4",
                                )
  
                            ],
                            # className="card-header"
                        ),    
                        dbc.Tooltip("Precision score ", target="precision",placement='bottom'),
                        dbc.Tooltip("Recall score", target="recall",placement='bottom'),
                        dbc.Tooltip("Performance of the model", target="auc",placement='bottom'),
                        html.Br(),   
                       
                     
                    ],
                    className="card-chart-contain col-md-12 col-lg-4",
                    id="cross-filter-options1",
                ),
                #--------------------------------------------------------------------------------------------
                html.Div(
                    [                                                 
                        html.Div(
                            [
                                html.Div(
                                    [dls.Hash(dcc.Graph(id="main_graph"),color="#435278",
                        speed_multiplier=2,
                        size=50,
                    ),],
                                    # className="card-chart-contain col-md-12 col-lg-7",
                                ),
                                # html.Div(
                                #     [dcc.Graph(id="aggregate_graph")],
                                #     className="card-chart-contain col-md-12 col-lg-4",
                                # ),                         

                            ],
                            className="row flex-display",
                        ),

                                        
                    ],
                    id="right-column",
                    className="col-md-12 col-lg-5",
                ),
               
            ],
            className="row flex-display",
        ),
         
        html.Hr(),
        html.Div(
            [
                html.Div(
                        [dls.Hash(dcc.Graph(id="fig_confusion", figure={}),color="#435278",
                        speed_multiplier=2,
                        size=50,
                    ),],
                        className="card-chart-contain col-md-12 col-lg-4",
                ), 
                html.Div(
                    [dls.Hash(dcc.Graph(id="fig_roc"),color="#435278",
                        speed_multiplier=2,
                        size=50,
                    ),],
                    className="card-chart-contain col-md-12 col-lg-4",
                ),
                html.Div(
                    [dls.Hash(dcc.Graph(id="fig_precision"),color="#435278",
                        speed_multiplier=2,
                        size=50,
                    ),],
                    className="card-chart-contain col-md-12 col-lg-4",
                ),
           
            
            ],
            className="row flex-display",
        ),
       
        html.Hr(),
        html.Div(
            [
    
               html.H3(
                                    "Prediction",
                                    style={"font-weight": "bold","color": "#0084d6",},
                                    className="text-center mb-4",

                                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin": "10px"},
        ),
        html.Div([ 
            html.Div(
                    [
                        
                        # html.P(lab_var[j]),
                    NamedGroup(
                        dcc.Dropdown(
                            id=j+'_id',
                            options = [{'label':x, 'value':x} for x in df[j].unique()],
                            value = df[j].unique()[0],
                            # multi=True,
                            clearable=False,
                            # className="dcc_control",

                        ),
                        label=lab_var[j]  
                    )for j in Personals+Contracts #+ categorical_feat
                        # html.Br(),
                    ] ,style={"color":"#0a66c2",},
                    className="col-md-12 col-lg-2",

                ),
            html.Div(
                    [
                        
                        # html.P(lab_var[j]),
                    NamedGroup(
                        dcc.Dropdown(
                            id=j+'_id',
                            options = [{'label':x, 'value':x} for x in df[j].unique()if x!="No P-S"],
                            value = "No",
                            # multi=True,
                            clearable=False,
                            # className="dcc_control",

                        ),
                        label=lab_var[j]  
                    )for j in [ "PhoneService","MultipleLines", "InternetService"]
                        # html.Br(),
                    ] +[
                        NamedGroup(
                            daq.NumericInput(
                                            id='tenure_id',
                                            min=0,
                                            max=73,
                                            size = 75,
                                            value=2
                                        ),  
                                
                            label=lab_var['tenure'],
                            # id="control-item-tenure",
                            ),
                        
                        NamedGroup(
                            # dbc.Input(placeholder="between 18$ and 120$", className="mb-3",id="MonthlyCharges_id"),
                            dcc.Slider(
                                id="MonthlyCharges_id",
                                min=18,
                                max=120,
                                value=35,
                                step=0.1,
                                tooltip={"placement": "bottom"},
                                marks={i: str(i) for i in [18,50,100,120]},
                            ),
                                                    
                            label=lab_var['MonthlyCharges'],
                            # id="control-item-tenure",
                        ),
                       html.Div([
                        html.I(className="menu-icon tf-icons bx bx-info-circle",id='icon_id'),
                        ],className="text-center"
                        ),
                       html.Div([dbc.Tooltip("Info", target="icon_id",placement='bottom')], className="text-center"),
                       dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Information")),
                                    dbc.ModalBody(
                                        dcc.Markdown('''
                                        - Once [No] is selected for `Phone Service` and `Internet Service`, the options for `Multiple Lines` and other `Internet-related services` are automatically disabled and [No P-S] and [No I-S] are automatically selected, respectively.

                                        - The variable `Total Charges` was not included in any model due to its high correlation with `Tenure` and `Monthly Charges`  (TotalCharges` = `tenure` x `MonthlyCharges`). 

                                        - Click on `Submit` to predict churn based on the selected model and input information. The model selected at the top will be used for the prediction.

                                        - Note  😉😉:

                                            To improve recall score, consider using a model such as Random Forest with an under-sampling technique
                                        
                                                                               
                                        ''')
                                    ),
                                ],
                                id="info-modale",
                                size="lg",
                                is_open=False,
                                # contentClassName='modalcontent1',
                                ),

                        html.Br(),
                        html.Hr(),
                        html.Div([dbc.Button("Submit",id="submit",n_clicks=0)], className="text-center"),

                    ],
                    style={"color":"#0a66c2",},
                    className="col-md-12 col-lg-2",

                ),
            html.Div(
                    [
                        
                        # html.P(lab_var[j]),
                    NamedGroup(
                        dcc.Dropdown(
                            id=j+'_id',
                            options = [{'label':x, 'value':x} for x in df[j].unique() if x!="No I-S"],
                            value = "No",
                            # multi=True,
                            clearable=False,
                            # className="dcc_control",

                        ),
                        label=lab_var[j]  
                    )for j in Services
                    
                        # html.Br(),
                    ] ,
                    style={"color":"#0a66c2",},
                    className="col-md-12 col-lg-2",

                ),
                html.Div(
                    [
                        # dbc.Button("Random Sample", id="random-train", style={"margin-top": "20px"}, n_clicks=0),
                    #    html.I(className="menu-icon tf-icons bx bx-info-circle",id='icon_id'),
                       
                       html.Div([
                        html.A(
                                html.Img(
                                    id="status_icon_id",
                                    # n_clicks=0,
                                    # src=,
                                    height="150px",
                                    # style={"margin": "14px"},
                                ),
                            ),
                       ],),
                       html.Div(id="status_id"),
                       html.Div(
                        [
                        daq.GraduatedBar(
                            id='model-graduated-bar',
                            label="Churn Prediction Probability",
                            max = 100,
                            color={"ranges":{"green":[0,40],"yellow":[40,50],"red":[50,100]}},
                            showCurrentValue=True,
                            value=0
                        ) ,   
                       
                            ],style={"color":"#0a66c2",},
                            # className="card-chart-contain col-md-12 col-lg-3",
                        ),
                    ] ,
                    style={'margin-top':100},
                    className="col-md-12 col-lg-6 text-center",

                )   
            ],
            className="row flex-display",
            )
    ],
    # id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

#----------------------------------------- Background color -------------------------------
@callback(
   Output("info-modale", "content_class_name"),
   Input("color_bg", 'data'),
)

def toggle_theme(data):
    if str(data)=="#fff":
        th = 'modalcontent1'
    else:
        th = 'modalcontent2'
    return th



# ---------------------------------------- Clear filter -------------------------------------------
@callback(Output("info-modale", "is_open"),
    Input("icon_id", "n_clicks"),
    State("info-modale", "is_open")
)
def toggle_modal(n_clicks, is_open):
    if (n_clicks):
        return not is_open
    return is_open

@callback(
   
    [
        Output(j+"_id", "disabled") for j in Services+['MultipleLines']
    # ]
    # +
    # [
    #     Output(j+"_id", "option") for j in Services+['MultipleLines']
    ],
    [
        Input("InternetService_id",'value'),
        Input("PhoneService_id",'value')

    ],
    # [
    #     State(j+"_id", "option") for j in Services+['MultipleLines']
    # ]
)
def update_check( InternetService_id,PhoneService_id
# ,DeviceProtection_id,OnlineBackup_id,TechSupport_id,StreamingTV_id,
#                     OnlineSecurity_id, StreamingMovies_id,MultipleLines_id
                    ):
    inactive1=False
    inactive2=False
    if InternetService_id=="No":
        inactive1=True
    if PhoneService_id=="No":
        inactive2=True
    disabled = [inactive1]*len(Services) + [inactive2]
    return disabled

@callback(
    [
        Output("records", "value"),        
        Output("trainset", "value"),
        Output("testset", "value"),
        Output("acc_baseline", "value"),
        Output("acc_train", "value"),
        Output("acc_test", "value"),
        Output("precision", "value"),
        Output("recall", "value"),
        Output("auc", "value"),
        Output("main_graph", "figure"),
        # Output('model-graduated-bar', 'value'),
        Output("fig_confusion", "figure"),
        Output("fig_roc", "figure"),
        Output("fig_precision", "figure"),

        # Output("categorical", "value")
       
    ],           
    [
        Input("select_sample", "value"),
        Input("slider",'value'),
        Input("select_models",'value')
    ]
)
def update_text( sampling,slider,clf):
    # split df_proc in feature matrix and target vector
    X=df_proc.drop(columns=['Churn','TotalCharges'])
    y=df_proc['Churn']

    test_size=(100-int(slider))/100
    # split df_proc between train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

    if sampling == "Under":
        rus = RandomUnderSampler()
        X_train, y_train= rus.fit_resample(X_train, y_train)
    elif sampling =="Over":
        ros = RandomOverSampler()
        X_train, y_train= ros.fit_resample(X_train, y_train)

    acc_baseline = round(y.value_counts(normalize=True).max(), 3)
    
    # # standardizing X_train and X_test
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Build model
    model = models_dict[clf]

    # Fit model to training data
    model.fit(X_train,y_train)

    acc_train = round(accuracy_score(y_train,model.predict(X_train)), 3)
    acc_test = round(model.score(X_test,y_test), 3)

    y_pred = model.predict(X_test)
    # precision tp / (tp + fp)
    precision = round(precision_score(y_test, y_pred),3)
    # recall: tp / (tp + fn)
    recall = round(recall_score(y_test, y_pred),3)
    # auc
    probs = model.predict_proba(X_test)[:, 1]
    lr_auc = round(roc_auc_score(y_test, probs),3)


    features = X.columns

    if clf=="Logistic" :
        importances = model.coef_[0] 
    elif clf != "SVC" :
        importances = model.feature_importances_
    else :
        importances = [0]*len(X.columns) 


    features_df = pd.Series(importances,index=features).sort_values()
    fig = px.histogram(y=features_df.tail(10).index,x=features_df.tail(10).values,orientation="h")
    fig.update_layout(
                title=f'<b>Features Importance in Churn Prediction<b>',
                xaxis_title=f'<b> <b>',
                yaxis_title='<b>Features<b>',
                font={'size': 12, 'family':'garamond','color':'black'},
            
                    title_x=0.5,
                    template='simple_white',
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgb(0,0,0,0)",
                    font_color="#909090",
                )


    df_corr = confusion_matrix(y_test, y_pred, normalize='true')
    # print(df_corr)
    # fig_corr = ff.create_annotated_heatmap(df_corr,x=[0,1],y=[0,1])
    fig_corr=px.imshow(df_corr,text_auto=True,
                labels=dict(x="Predicted Label",y="True Label"),
                y=["0","1"], x=["0","1"])
    fig_corr.update_layout(
                title=f'Confusion Matrix',

                font={'size': 12, 'family':'garamond','color':'black'},
            
                    title_x=0.5,
                    title_y=.9,

                    template='simple_white',
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgb(0,0,0,0)",
                    font_color="#909090",
                )

    lr_fpr, lr_tpr, thresholds = roc_curve(y_test, probs)     
        
    lr_auc = round(roc_auc_score(y_test, probs),2)
    fig_ROC = px.area(
        x=lr_fpr, y=lr_tpr,
        title=f'ROC Curve (AUC={lr_auc:.4f})',
    
        labels=dict(x='False Positive Rate', y='True Positive Rate')
    
    )
    fig_ROC.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig_ROC.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_ROC.update_xaxes(constrain='domain')
    fig_ROC.update_layout(
                font={'size': 12, 'family':'garamond','color':'black'},
            
                    title_x=0.5,
                    template='simple_white',
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgb(0,0,0,0)",
                    font_color="#909090",
                )


    fig_precision = px.histogram(
        x = probs, color=y_test, nbins=50,
        labels=dict(color='True Labels', x='Score')
    )

    fig_precision.update_layout(
                font={'size': 12, 'family':'garamond','color':'black'},
            
                    title_x=0.5,
                    template='simple_white',
                    plot_bgcolor="rgb(0,0,0,0)",
                    legend_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgb(0,0,0,0)",
                    font_color="#909090",
                )

    
    return df_proc.shape[0], X_train.shape[0] , X_test.shape[0],acc_baseline,acc_train,acc_test,precision,recall,lr_auc,fig,fig_corr, fig_ROC,fig_precision



@callback(
    [
        Output("status_id","children"),
        Output('model-graduated-bar', 'value'),
        Output('status_icon_id', 'src'),

    ],
    [
        Input("select_sample", "value"),
        Input("slider",'value'),
        Input("select_models",'value')
    ]+[
        Input(j+"_id","value") for j in features_obj[:-2]
    ]+[
        Input("submit","n_clicks")
    ]
)

def prediction_churn(sampling,slider,clf, gender, SeniorCitizen, Partner,Dependents, tenure,
        PhoneService, MultipleLines, InternetService,
       OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
       StreamingTV, StreamingMovies, Contract, PaperlessBilling,
       PaymentMethod, MonthlyCharges,n_clicks):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'submit' in changed_id:
        
        # split df_proc in feature matrix and target vector
        X=df_proc.drop(columns=['Churn','TotalCharges'])
        y=df_proc['Churn']
        # print(df_proc.columns)
        test_size=(100-int(slider))/100
        # split df_proc between train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)

        if sampling == "Under":
            rus = RandomUnderSampler()
            X_train, y_train= rus.fit_resample(X_train, y_train)
        elif sampling =="Over":
            ros = RandomOverSampler()
            X_train, y_train= ros.fit_resample(X_train, y_train)

        
        # Build model
        model = models_dict[clf]

        # Fit model to training data
        model.fit(X_train,y_train)
        # print(gender)
        X_submit  = pd.DataFrame(columns=X.columns,index=[0])
        X_submit['gender'] = 1 if gender =="Male" else 0
        X_submit['SeniorCitizen'] = 1 if SeniorCitizen =="Yes" else 0
        X_submit['Dependents'] = 1 if Dependents  =="Yes" else 0
        X_submit['Partner'] = 1 if Partner  =="Yes" else 0
        X_submit['PaperlessBilling'] = 1 if PaperlessBilling  =="Yes" else 0
        X_submit['PhoneService'] = 1 if PhoneService  =="Yes" else 0
        X_submit['tenure'] = tenure 
        X_submit['MonthlyCharges'] = MonthlyCharges
        X_submit['MultipleLines_No'] = 1 if (MultipleLines  =="No" and PhoneService  =="Yes")  else 0
        X_submit['MultipleLines_Yes'] = 1 if (MultipleLines  =="Yes" and PhoneService  =="Yes")  else 0
        X_submit['MultipleLines_No P-S'] = 1 if  PhoneService  =="No" else 0
        X_submit['InternetService_No'] = 1 if InternetService  =="No" else 0
        X_submit['InternetService_DSL'] = 1 if InternetService  =="DSL" else 0
        X_submit['InternetService_Fiber optic'] = 1 if InternetService  =="Fiber optic" else 0
        X_submit['OnlineSecurity_No I-S'] = 1 if InternetService  =="No" else 0
        X_submit['OnlineBackup_No I-S'] = 1 if InternetService  =="No" else 0
        X_submit['DeviceProtection_No I-S'] = 1 if InternetService  =="No" else 0
        X_submit['TechSupport_No I-S'] = 1 if InternetService  =="No" else 0
        X_submit['StreamingTV_No I-S'] = 1 if InternetService  =="No" else 0
        X_submit['StreamingMovies_No I-S'] = 1 if InternetService  =="No" else 0
        X_submit['Contract_Month-to-month'] = 1 if Contract  =="Month-to-month" else 0
        X_submit['Contract_One year'] = 1 if Contract  =="One year" else 0
        X_submit['Contract_Two year'] = 1 if Contract  =="Two year" else 0
        X_submit['PaymentMethod_Bank transfer'] = 1 if PaymentMethod  =="Bank transfer" else 0
        X_submit['PaymentMethod_Credit card'] = 1 if PaymentMethod  =="Credit card" else 0
        X_submit['PaymentMethod_Electronic check'] = 1 if PaymentMethod  =="Electronic check" else 0
        X_submit['PaymentMethod_Mailed check'] = 1 if PaymentMethod  =="Mailed check" else 0
        X_submit['OnlineSecurity_No'] = 1 if (OnlineSecurity  =="No" and InternetService  =="Yes")  else 0
        X_submit['OnlineBackup_No'] = 1 if (OnlineBackup  =="No" and InternetService  =="Yes")  else 0
        X_submit['DeviceProtection_No'] = 1 if (DeviceProtection  =="No" and InternetService  =="Yes")  else 0
        X_submit['TechSupport_No'] = 1 if (TechSupport  =="No" and InternetService  =="Yes")  else 0
        X_submit['StreamingTV_No'] = 1 if (StreamingTV  =="No" and InternetService  =="Yes")  else 0
        X_submit['StreamingMovies_No'] = 1 if (StreamingMovies  =="No" and InternetService  =="Yes")  else 0
        X_submit['OnlineSecurity_Yes'] = 1 if (OnlineSecurity  =="Yes" and InternetService  =="Yes")  else 0
        X_submit['OnlineBackup_Yes'] = 1 if (OnlineBackup  =="Yes" and InternetService  =="Yes")  else 0
        X_submit['DeviceProtection_Yes'] = 1 if (DeviceProtection  =="Yes" and InternetService  =="Yes")  else 0
        X_submit['TechSupport_Yes'] = 1 if (TechSupport  =="Yes" and InternetService  =="Yes")  else 0
        X_submit['StreamingTV_Yes'] = 1 if (StreamingTV  =="Yes" and InternetService  =="Yes")  else 0
        X_submit['StreamingMovies_Yes'] = 1 if (StreamingMovies  =="Yes" and InternetService  =="Yes")  else 0
        
        yhat=model.predict(X_submit)[0]
        proba = model.predict_proba(X_submit)[0][1]
        cond = yhat==1
        color = 'red' if cond else 'green'
        name = 'Churned' if cond else 'Stay'
        status = html.P(name,style={"color": color,"font-weight": "bold","font-size": "300%","text-align": "center"},)
        icone = "./assets/images/dislike.ico" if cond else "./assets/images/like.ico"
        # print(proba,yhat)
        return status,proba*100,icone
    else :
        return None,0,None

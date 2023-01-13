import dash
from dash import html
import dash_bootstrap_components as dbc
from utils.consts import LINKEDIN_PROFILE
from dash import Dash, Input, Output, callback, dcc, html,State




dash.register_page(
    __name__,
    path='/',
    title="Customer Churn",
    name="Home",
    # image="metatag-image.png",
    description="Dash app developed for the Dash Autumn Challenge. The Customer Churn data was used."
)

layout = html.Div([
    html.Div([
            html.Div(
                className="col-md-12 col-sm-12 col-lg-12 mb-md-0 mb-4 card-chart-container", 
                children=[

                html.Div(className="card",
                children=[
                    dbc.Row([
                        dbc.Col(className="col-lg-6", children=[html.Div(className="card-header card-m-0 me-2 pb-3", children=[
                            html.H2(["Customer Churn Dashboard"],
                                # className="card-title m-0 me-2 mb-2",
                                 style={"font-size": "2vw"}),
                            html.Span(
                                "Dash Holliday Challenge", style={"color": "#0084d6", "font-size": "1.5vw"}
                            )
                        ]),
                            html.P([html.P("Welcome to your customer churn dashboard!",className="mt-1",),
                                   
                                    html.P(
                                        "Here, we analyze customer churn behavior, as well as use a classification model to predict which customers are at risk of churning. By understanding and addressing the root causes of churn, you can work to retain valuable customers and improve your business's revenue and profitability.",
                                        className="mt-1",style={'textAlign': 'justify'}),
                                    html.P(
                                        ["We use the customer churn dataset from " ,html.A("IBM", href="https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113"
                                        ,target="_blank",style={"color": "#0084d6"}),
                                         " provided by ",html.A("Plotly", href="https://plotly.com/",target="_blank",style={"color": "#0084d6"}),  " for this exercice"], className="mt-1")
                                    ], className="card-title me-4"),


                            html.P([html.P("If you have any questions or need assistance, please don't hesitate to reach out."),
                                    html.P(html.A(" Abdel An'lah Tidjani.", href=LINKEDIN_PROFILE, target="_blank", style={"color": "#0084d6"}))],
                                     className="card-title me-4 mb-0 mt-4"),

                        ]),

                        dbc.Col(
                            className="col-lg-6", 
                            children=[
                            html.Img(
                            src="./assets/images/ch.png", className="img-fluid"),
                            ]
                        ),
                    ]),
                ],
                id="wel",
                )
            ])
        ],
        
        ),

    

    
], style={"padding-top": "40px"}
)


@callback(
   Output("wel", "style"),
   Input("color_bg", 'data'),
)

def toggle_theme(data):
    return {'background-color': str(data)}
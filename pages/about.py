
import dash
from dash import html
import utils.theme as theme
from dash import Dash, Input, Output, callback, dcc, html,State

# from utils.consts import GITHUB_PROFILE, LINKEDIN_PROFILE, FACEBOOK_PROFILE

dash.register_page(
    __name__,
    path='/about',
    title="Customer Churn",
)
about_me_text= "Graduate in economics with a love for programming. Using my knowledge of economics and programming to help businesses and organizations identify new opportunities, optimize their operations, and make better decisions."
layout = html.Div(className="col-md-12 col-sm-12 col-lg-8 mb-md-0 mb-4 card-chart-container",
    children=[html.Div( className="card",
    children=[
        html.Div(className="card-body p-0", children=[
            html.Div(className="d-flex justify-content-between", children=[
                html.Div(className="card-info p-4 ",
                         children=[html.H3(className="card-text", children=["Who am I?"]),
                                    html.H2(className="card-text m-0 p-0", children=["Abdel An'lah Tidjani"] , style={"color":theme.COLOR_PALLETE[0]}),
                                   html.Div(className="mb-2 mt-2", children=[
                                       html.P(className="card-title mb-2",
                                            children=[about_me_text], style={"font-size":"1rem"}),
                                   ]),
                                   html.Small(
                             className="card-text", children=[]),
                             html.A(href="https://www.linkedin.com/in/abdelanlah-tidjani/",target="_blank" ,children=[
                                html.I(className="bx bxl-linkedin-square mt-3", style={"font-size":"2.5rem" , "color":"#0a66c2"}),]),
                             html.A(href='https://community.plotly.com/u/abdelanlah/summary',target="_blank",
                             children=[
                                html.A(html.Img(src="./assets/images/plotly_bg.ico", height="30px",style={"margin-top":15, "background-color":"#0a66c2"},),),
                                ]),

                             html.A(href="https://github.com/AbdelTID/",target="_blank",
                             children=[html.I(className="bx bxl-github mt-3" , style={"font-size":"2.5rem" , "color":"#0a66c2"})]),

                            #  html.Br(),
                            #  html.Br(),
                            #  html.Br(),

                            #  html.H3(className="card-text", children=["Reference"]),
                            #  html.Div(className="mb-2 mt-2", children=[
                            #                      html.P(
                            #             ["Template originaly come from " ,html.A("Ivan Abboud", href="https://www.linkedin.com/in/ivan-abboud-737b2120a/",target="_blank",style={"color": "#0084d6"}),
                            #              " with is project ",html.A("Fifa Worldcup Dashboard", href="http://ivan96.pythonanywhere.com/",target="_blank",style={"color": "#0084d6"}),],
                            #               className="card-title me-4 mb-0 mt-4")
                            #        ]),

                         ]),
                # html.Div(className="card-icon d-flex align-items-end", children=[
                #     # html.Img(className="img-fluid",
                #     #          src="./assets/images/programmer.gif" , style={"border-radius":6})
                # ]
                # )
            ])

        ])
    ],
    id='about_id'),
    ]
    )

@callback(
   Output("about_id", "style"),
   Input("color_bg", 'data'),
)

def toggle_theme(data):
    return {'background-color': str(data)}
import sys
import os
from dash.dependencies import Input, Output,State

module_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from components.NavbarVertical import sidebar
from components.Footer import Footer

import glob


# # RAW

ROOT_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))
SRC_FOLDER = os.path.join(ROOT_FOLDER, "src/")
ASSETS_FOLDER = os.path.join(SRC_FOLDER, "assets")



external_style_sheet = glob.glob(os.path.join(
    ASSETS_FOLDER, "bootstrap/css") + "/*.css")
external_style_sheet += glob.glob(os.path.join(ASSETS_FOLDER,
                                  "css") + "/*.css")
external_style_sheet += glob.glob(os.path.join(ASSETS_FOLDER,
                                  "fonts") + "/*.css")


app = dash.Dash(__name__, title="Customer Churn Dashboard",use_pages=True,
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP] + external_style_sheet,
                suppress_callback_exceptions=True,
                )
color_bg_store = html.Div([dcc.Store(id="color_bg", data="#f8f9fa"),])



server = app.server


app.layout = html.Div(id='app',className="layout-wrapper layout-content-navbar",
                      children=[
                          html.Div(className="layout-container",
                                   children=[
                                       dcc.Location(id="url"),
                                       color_bg_store,
                                       html.Aside(className="",
                                                  children=[
                                                      sidebar

                                                  ]),
                                       html.Div(className="layout-page",
                                                children=[
                                                    html.Div(className="content-wrapper",
                                                             children=[
                                                                 html.Div(className="container-xxl flex-grow-1 container-p-y p-0",
                                                                        #   id="page-content",
                                                                          children=[dash.page_container,

                                                                          ]),
                                                                 html.Footer(className="content-footer footer bg-footer-theme",
                                                                             children=[
                                                                                 Footer#,dark_theme_button
                                                                             ], style={"margin-left": "6rem"})

                                                             ])
                                                ])

                                   ])
                      ])


#
@app.callback(
   [ Output('app', 'style'),
    Output("color_bg", 'data'),
    Output('theme-btn', 'src'), 
    Output('L_tip', 'children'),
    Output("bg_id", "style")],
    Input('theme-btn', 'n_clicks')
)

def toggle_theme(n_clicks):
    if n_clicks%2:
        app.css.config.external_stylesheets = [dbc.themes.DARKLY]
        th = {'background-color': '#222430'}
        return {'background-color':"#111111" , 'color': '#f7f7f7'} , '#222430',"./assets/images/sun.svg","Light mode", th
    else:
        app.css.config.external_stylesheets = [dbc.themes.BOOTSTRAP]
        th ={'background-color': "#fff"}
        return {'background-color': '#f5f5f9', 'color': '#222430'},"#fff","./assets/images/moon.svg","Dark mode", th


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)

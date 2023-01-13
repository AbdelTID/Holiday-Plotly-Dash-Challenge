from dash import html
import dash_bootstrap_components as dbc



sidebar = html.Div(
    [
        html.Div(
            [
                html.Img(src="./assets/images/ch.png", style={"width": "3rem"}),
                html.H4("Customer Churn", className="m-0"),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="menu-icon tf-icons bx bx-home"), html.Span("Welcome")],
                    href="/",
                    active="exact",
                    className="pe-3"
                ),
                dbc.NavLink(
                    [
                        html.I(className="menu-icon tf-icons bx bxs-contact"),
                        html.Span("Customers"),
                    ],
                    href="/explore",
                    active="exact",
                    className="pe-3"
                ),
                dbc.NavLink(
                    [
                        html.I(className="menu-icon tf-icons bx bx-bulb"),
                        html.Span("Insights"),
                    ],
                    href="/call",
                    active="exact",
                    className="pe-3"
                ),
                dbc.NavLink(
                    [
                        html.I(className="menu-icon tf-icons bx bx-bar-chart-square"),
                        html.Span("Exploratory"),
                    ],
                    href="/EDA",
                    active="exact",
                    className="pe-3"
                ),
                
                dbc.NavLink(
                    [
                        html.I(className="menu-icon tf-icons bx bx-analyse"),
                        html.Span("Classification"),
                    ],
                    href="/classification1",
                    active="exact",
                    className="pe-3"
                ),
                dbc.NavLink(
                    [
                        html.I(className="menu-icon tf-icons bx bxs-file"),
                        html.Span("Summary"),
                    ],
                    href="/summary",
                    active="exact",
                    className="pe-3",
                ),
                dbc.NavLink(
                    [
                        html.I(className="menu-icon tf-icons bx bx-info-circle"),
                        html.Span("About"),
                    ],
                    href="/about",
                    active="exact",
                    className="pe-3",
                ),
                
                dbc.NavItem(html.A(
                                html.Img(
                                    id='theme-btn',
                                    n_clicks=0,
                                    src="./assets/images/moon.svg", height="20px",
                                    style={"margin": "14px"},
                                ),
                            ),
                        ),
                dbc.Tooltip("Dark Theme", target="theme-btn",placement='right',id='L_tip',),                
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="bg_id",
    className="sidebar",
)


import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path='/summary',
    title="Documentation",
    name="Summary"
)

layout = dbc.Container(
    dbc.Row([ 
        dbc.Col([ 
            html.Br(), html.Br(),
            html.H3("App Toc"),
            html.Hr(),

            dcc.Markdown('''
            
            
            The app consists of seven sections:
            - `Welcome`: Welcome page.
            - `Customers`: Display each customer information.
            - `Insights`: This section gives an overview of customers behavior. 
            - `Exploratory`: This section shows relationship between churn and monthly charges within customer caracteristiques.
            - `Classification`: This section provides 
            - `Summary`: Give a summary of the app. You are here 😁😄😅.
            - `About`: Who I am?.

            '''),

        #     html.Br(),

        #     html.H3("Data Dictionary"),

        #     html.Hr(),

        #     dcc.Markdown('''
            
        # **Demographics**
        #   * `CustomerID`- A unique ID that identifies each customer.
        #   * `Gender`- The customer's gender- [Male, Female]
        #   * `Senior Citizen`- Indicates if the customer is 65 or older- [Yes, No]
        #   * `Partner`- Indicates if the customer is married- [Yes, No]
        #   * `Dependents`- Indicates if the customer lives with any dependents- [Yes, No]. Dependents could be children, parents, grandparents, etc.
        
        # **Services**
        #   * `Tenure in Months`- Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
        #   * `Phone Service`- Indicates if the customer subscribes to home phone service with the company- [Yes, No]
        #   * `Multiple Lines`- Indicates if the customer subscribes to multiple telephone lines with the company- [Yes, No]
        #   * `Internet Service`- Indicates if the customer subscribes to Internet service with the company- [No, DSL, Fiber Optic].
        #   * `Online Security`- Indicates if the customer subscribes to an additional online security service provided by the company- [Yes, No]
        #   * `Online Backup`- Indicates if the customer subscribes to an additional online backup service provided by the company- [Yes, No]
        #   * `Device Protection Plan`- Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company- [Yes, No]
        #   * `Tech Support`- Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times- [Yes, No]
        #   * `Streaming TV`- Indicates if the customer uses their Internet service to stream television programing from a third party provider- [Yes, No]. The company does not charge an additional fee for this service.
        #   * `Streaming Movies`- Indicates if the customer uses their Internet service to stream movies from a third party provider- [Yes, No]. The company does not charge an additional fee for this service.
        #   * `Contract`- Indicates the customer's current contract type-[ Month-to-Month, One Year, Two Year].
        #   * `Paperless Billing`- Indicates if the customer has chosen paperless billing- [Yes, No]
        #   * `Payment Method`- Indicates how the customer pays their bill-  ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
        #   * `Monthly Charge`- Indicates the customer's current total monthly charge for all their services from the company.
        #   * `Total Charges`- Indicates the customer's total charges, calculated to the end of the quarter specified above.
        
        # **Status** Outcomes
        #   * `Churn Label`- [Yes] = the customer left the company this quarter. [No] = the customer remained with the company. Directly related to Churn Value.
            
        #     '''),

            html.Br(),

            html.H3("Executive Summary"),

            html.Hr(),

            dcc.Markdown('''
           Our analysis of customer churn data has identified several trends and patterns that may be contributing to churn. Customers who are on month-to-month contracts and those who are dissatisfied with their experience with our company are more likely to churn. Additionally, customers who pay their bills using electronic checks or automatic bank transfers are less likely to churn. By implementing the following recommendations, we believe we can reduce customer churn and improve customer retention:
           
           '''),

                     html.Br(),

            html.H5("Key Findings:"),

            html.Hr(),

            dcc.Markdown('''
            - Customers on month-to-month contracts are more likely to churn
            - 25% of churned customers leave within the first two months, while 50% leave within the first 10 months.
            - Customers who are dissatisfied with their experience are more likely to churn
            - Customers who pay their bills using electronic checks or automatic bank transfers are less likely to churn
           '''),
           html.Br(),

            html.H5("Recommendations:"),

            html.Hr(),

            dcc.Markdown('''
          - Offer incentives for customers to sign longer-term contracts
          - Consider offering three month contracts to potentially reduce rapid customer churn and retain them for a longer period of time.
          - Invest in customer service training and improve service offerings to increase customer satisfaction
          - Offer more flexible payment options, such as electronic checks and automatic bank transfers
           '''),

          html.Br(),

            html.H5("Next Steps:"),

            html.Hr(),

            dcc.Markdown('''
          - Conduct a pilot program to test the effectiveness of offering incentives for customers to sign longer-term contracts
          - Review and update customer service training and service offerings based on customer feedback
          - Implement electronic check and automatic bank transfer payment options
           '''),
                     html.Br(),

            html.H5("Appendices:"),

            html.Hr(),

            dcc.Markdown(link_target="_blank" ,
           children=['''
          - [Customer churn data](https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn-by-IBM.csv)
          - [Data description](https://docs.google.com/document/d/1iazD22swSw6CJAVDwkVk-Ynh-r-4meWclfQ0jVMMubU/edit)
           ''']),
          
          html.Br(),

            html.H5("References:"),

            html.Hr(),

            dcc.Markdown(link_target="_blank" ,
           children=[ '''
           - Template originaly come from [Ivan Abboud](https://www.linkedin.com/in/ivan-abboud-737b2120a/) with is project [Fifa Worldcup Dashboard](http://ivan96.pythonanywhere.com/)
           - Insights session adapted from:  https://www.inetsoft.com/evaluate/bi_visualization_gallery/dashboard.jsp?dbIdx=1 
           ''']
           )
        ])
    ])
)

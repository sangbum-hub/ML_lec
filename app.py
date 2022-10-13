# pip install --user dash
# pip install --user dash_bootstrap_components
import dash 
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc 

# Dash 함수를 불러와서 Web 구성 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
############################################################################################
# 데이터를 처리하는 부분 
import pandas as pd 
import plotly.express as px 

df1 = pd.read_csv('01_Data.csv')
p1 = df1.pivot_table(index='Channel', values='Amount_Month',aggfunc='sum').reset_index()

fig1 = px.bar(p1, x='Channel', y='Amount_Month')
############################################################################################
# Callback 
from dash.dependencies import Input, Output
import pickle 

model = pickle.load(open('model_web.sav','rb'))

@app.callback(
    Output('result', 'children'),
    Input('Term','value'),
    Input('Product_Type','value'),
    Input('Amount_Month','value'),
    Input('Age','value'),
    Input('Gender','value'),
    Input('Credit_Rank','value'),
)
def predict_function(x1,x2,x3,x4,x5,x6):
    input_list = [x1,x2,x3,x4,x5,x6]
    column_list = ['Term', 'Product_Type', 'Amount_Month','Age','Gender','Credit_Rank']
    input_data = pd.DataFrame(data= [input_list], columns=column_list )

    result = model.predict_proba(input_data)[0][1]

    return f'{result:.2f}%'

############################################################################################
# Web이 표시되는 View Layout 
input_layout = [
    html.Div([
        html.Label('계약 기간'),
        dcc.Input(id='Term', type='text')
    ]),
    html.Div([
        html.Label('제품군'),
        dcc.Input(id='Product_Type', type='text')
    ]),
    html.Div([
        html.Label('월랜탈비용'),
        dcc.Input(id='Amount_Month', type='text')
    ]),
    html.Div([
        html.Label('고객 연령'),
        dcc.Input(id='Age', type='text')
    ]),
    html.Div([
        html.Label('고객 성별'),
        dcc.Input(id='Gender', type='text')
    ]),
    html.Div([
        html.Label('신용 등급'),
        dcc.Input(id='Credit_Rank', type='text')
    ])    
]

app.layout = dbc.Container(
    [
        html.H3(children='Data Web DashBoard'),
        dbc.Col(html.Div(input_layout)),
        html.Hr(),
        dbc.Card([
            html.H2(id='result', className='card_title'),
            html.P('해약 확률', className='card_text')
        ]),
        dbc.Col(children=[ html.Div(dcc.Graph(figure= fig1)) ])
    ]
)
############################################################################################
# 구성된 Web을 구동 
if __name__ == '__main__':
    app.run_server(debug=True)


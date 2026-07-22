# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load data
data = pd.read_csv('data.csv')

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='example-graph'),
    dcc.Input(id='input', value='initial value', type='text')
])

# Callback to update the graph based on input
@app.callback(
    Output('example-graph', 'figure'),
    [Input('input', 'value')]
)
def update_graph(input_value):
    fig = px.scatter(data, x=input_value, y='column_name')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
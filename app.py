from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

df = pd.read_csv('brain_stroke.csv')
df['stroke'] = df['stroke'].map({0: 'Brak udaru', 1: 'Udar'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard_pie')
def dashboard_pie():
    fig = px.pie(df, names='stroke', title='Rozkład przypadków udaru')
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_pie.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)

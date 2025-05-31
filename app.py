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

@app.route('/dashboard_age')
def dashboard_age():
    fig = px.histogram(df, x='age', color='stroke', nbins=30, 
                       title='Rozkład wieku względem udaru',
                       labels={'age': 'Wiek', 'count': 'Liczba przypadków'})
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_age.html', graph_html=graph_html)

@app.route('/dashboard_scatter')
def dashboard_scatter():
    fig = px.scatter(df, x='avg_glucose_level', y='bmi', color='stroke',
                     title='Poziom glukozy a BMI',
                     labels={'avg_glucose_level': 'Średni poziom glukozy', 'bmi': 'BMI'})
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_scatter.html', graph_html=graph_html)

@app.route('/dashboard_cm')
def dashboard_cm():
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        z=corr_matrix.values,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_cm.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)

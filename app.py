from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

app = Flask(__name__)

df = pd.read_csv('brain_stroke.csv')
df_mapped = df.copy()
df_mapped['stroke'] = df_mapped['stroke'].map({0: 'Brak udaru', 1: 'Udar'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard_stats')
def dashboardstats():
  numeric_df = df.select_dtypes(include=['number'])
  categorical_df = df_mapped.select_dtypes(include=['object', 'category', 'bool'])

  numeric_stats = {}
  for column in numeric_df.columns:
      numeric_stats[column] = {
          'mean': numeric_df[column].mean(),
          'median': numeric_df[column].median(),
          'min': numeric_df[column].min(),
          'max': numeric_df[column].max(),
          'std': numeric_df[column].std(),
          'missing': numeric_df[column].isna().sum()
      }

  categorical_stats = {}
  for column in categorical_df.columns:
      categorical_stats[column] = {
          'unique': categorical_df[column].nunique(),
          'mode': categorical_df[column].mode().iloc[0] if not categorical_df[column].mode().empty else None,
          'top_freq': categorical_df[column].value_counts().iloc[0] if not categorical_df[column].value_counts().empty else 0,
          'missing': categorical_df[column].isna().sum()
      }

  return render_template('dashboard_stats.html', numeric_stats=numeric_stats, categorical_stats=categorical_stats)

@app.route('/dashboard_pie')
def dashboard_pie():
    gender = request.args.get('gender', 'both')
    min_age = math.floor(float(request.args.get('min_age', str(df['age'].min()))))
    max_age = math.floor(float(request.args.get('max_age', str(df['age'].max()))))

    filtered_df = df_mapped.copy()
    if gender in ['Male', 'Female']:
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]

    gender_label = gender if gender != "both" else "Wszyscy"
    title = f'Rozkład udarów - Wiek: {min_age}-{max_age}, Płeć: {gender_label}'

    fig = px.pie(
        filtered_df, 
        names='stroke', 
        title=title,
    )

    template_data = {
        'graph_html': fig.to_html(full_html=False),
        'current_gender': gender,
        'min_age': min_age,
        'max_age': max_age,
        'age_range': [df['age'].min(), df['age'].max()]
    }

    return render_template('dashboard_pie.html', **template_data)

@app.route('/dashboard_age')
def dashboard_age():
    fig = px.histogram(df_mapped, x='age', color='stroke', nbins=30, 
                       title='Rozkład wieku względem udaru', height=650,
                       labels={'age': 'Wiek', 'count': 'Liczba przypadków'})
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_age.html', graph_html=graph_html)

@app.route('/dashboard_scatter')
def dashboard_scatter():
    fig = px.scatter(df_mapped, x='avg_glucose_level', y='bmi', color='stroke', 
                     title='Poziom glukozy a BMI', height=650,
                     labels={'avg_glucose_level': 'Średni poziom glukozy', 'bmi': 'BMI'})
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_scatter.html', graph_html=graph_html)

@app.route('/dashboard_work')
def dashboard_work():
    fig = px.histogram(df_mapped, x='work_type', color='gender', facet_col='stroke', height=650,
                       title='Rodzaj pracy i płeć względem udaru')
    graph_html = fig.to_html(full_html=False)
    return render_template('dashboard_work.html', graph_html=graph_html)

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

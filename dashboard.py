import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import plotly.express as px
import plotly.graph_objects as go
import os

app = Flask(__name__)

# Load weather data
df = pd.read_csv('weatherAUS.csv')

# Data preprocessing from notebook
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['season'] = df['month'] % 12 // 3 + 1
df['weekday'] = df['Date'].dt.dayofweek

# Create derived features
df['CloudMean'] = df[['Cloud9am','Cloud3pm']].mean(axis=1)
df['TempMean'] = (df['MinTemp'] + df['MaxTemp']) / 2
df['TempRange'] = df['MaxTemp'] - df['MinTemp']
df['HumidityMean'] = (df['Humidity9am'] + df['Humidity3pm']) / 2
df['TempAvg'] = (df['Temp9am'] + df['Temp3pm']) / 2
df['WindSpeedMean'] = df[['WindSpeed9am','WindSpeed3pm','WindGustSpeed']].mean(axis=1)

# Map RainTomorrow and RainToday
df['RainTomorrowLabel'] = df['RainTomorrow']
df['RainTomorrow'] = df['RainTomorrow'].map({'No':0,'Yes':1})

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Weather Dashboard</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #f5f5f5; }
        h1 { text-align: center; }
        .chart { background: white; margin: 20px auto; padding: 10px; max-width: 1200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1>Weather Australia Analysis</h1>
    <div class="chart">{{ chart1 | safe }}</div>
    <div class="chart">{{ chart2 | safe }}</div>
    <div class="chart">{{ chart3 | safe }}</div>
    <div class="chart">{{ chart4 | safe }}</div>
    <div class="chart">{{ chart5 | safe }}</div>
    <div class="chart">{{ chart6 | safe }}</div>
    <div class="chart">{{ chart7 | safe }}</div>
    <div class="chart">{{ chart8 | safe }}</div>
    <div class="chart">{{ chart9 | safe }}</div>
    <div class="chart">{{ chart10 | safe }}</div>
    <div class="chart">{{ chart11 | safe }}</div>
    <div class="chart">{{ chart12 | safe }}</div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    # 1. TempMean vs RainTomorrow - Box plot
    fig1 = px.box(df, x='RainTomorrowLabel', y='TempMean', 
                  title='Temperature Mean vs Rain Tomorrow',
                  color='RainTomorrowLabel',
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  labels={'RainTomorrowLabel': 'Rain Tomorrow', 'TempMean': 'Temperature Mean (°C)'})
    
    # 2. HumidityMean vs RainTomorrow - Box plot
    fig2 = px.box(df, x='RainTomorrowLabel', y='HumidityMean',
                  title='Humidity Mean vs Rain Tomorrow',
                  color='RainTomorrowLabel',
                  color_discrete_sequence=px.colors.qualitative.Pastel,
                  labels={'RainTomorrowLabel': 'Rain Tomorrow', 'HumidityMean': 'Humidity Mean (%)'})
    
    # 3. Evaporation Distribution by RainTomorrow
    fig3 = px.histogram(df, x='Evaporation', color='RainTomorrowLabel', 
                       nbins=30, barmode='overlay', opacity=0.6,
                       title='Evaporation Distribution by Rain Tomorrow',
                       color_discrete_sequence=['#87CEEB', '#FF8C00'],
                       labels={'Evaporation': 'Evaporation (mm)', 'RainTomorrowLabel': 'Rain Tomorrow'})
    
    # 4. TempAvg Distribution by RainTomorrow
    fig4 = px.histogram(df, x='TempAvg', color='RainTomorrowLabel',
                       nbins=30, barmode='overlay', opacity=0.6,
                       title='Temperature Average Distribution by Rain Tomorrow',
                       color_discrete_sequence=['#90EE90', '#FA8072'],
                       labels={'TempAvg': 'Temperature Average (°C)', 'RainTomorrowLabel': 'Rain Tomorrow'})
    
    # 5. RainTomorrow Distribution by Location
    location_counts = df.groupby(['Location', 'RainTomorrowLabel']).size().reset_index(name='Count')
    fig5 = px.bar(location_counts, x='Location', y='Count', color='RainTomorrowLabel',
                  barmode='group', title='Rain Tomorrow Distribution by Location',
                  color_discrete_sequence=px.colors.qualitative.Bold,
                  labels={'RainTomorrowLabel': 'Rain Tomorrow'})
    fig5.update_layout(xaxis_tickangle=45, height=600)
    
    # 6. RainTomorrow Distribution by Day of Month
    day_counts = df.groupby(['day', 'RainTomorrowLabel']).size().reset_index(name='Count')
    fig6 = px.bar(day_counts, x='day', y='Count', color='RainTomorrowLabel',
                  barmode='group', title='Rain Tomorrow Distribution by Day of Month',
                  color_discrete_sequence=px.colors.qualitative.Safe,
                  labels={'day': 'Day of Month', 'RainTomorrowLabel': 'Rain Tomorrow'})
    
    # 7. RainTomorrow Distribution by Season
    season_counts = df.groupby(['season', 'RainTomorrowLabel']).size().reset_index(name='Count')
    season_map = {1: 'Summer', 2: 'Fall', 3: 'Winter', 4: 'Spring'}
    season_counts['season_label'] = season_counts['season'].map(season_map)
    fig7 = px.bar(season_counts, x='season_label', y='Count', color='RainTomorrowLabel',
                  barmode='group', title='Rain Tomorrow Distribution by Season',
                  color_discrete_sequence=px.colors.qualitative.Vivid,
                  labels={'season_label': 'Season', 'RainTomorrowLabel': 'Rain Tomorrow'})
    
    # 8. Average Evaporation per Weekday
    evap_weekday = df.groupby('weekday')['Evaporation'].mean().reset_index()
    weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    evap_weekday['weekday_label'] = evap_weekday['weekday'].map(weekday_map)
    fig8 = px.bar(evap_weekday, x='weekday_label', y='Evaporation',
                  title='Average Evaporation per Weekday',
                  labels={'weekday_label': 'Weekday', 'Evaporation': 'Evaporation (mm)'},
                  color='Evaporation', color_continuous_scale='teal')
    
    # 9. Average Evaporation per Month
    month_evap = df.groupby('month')['Evaporation'].mean().reset_index()
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_evap['month_label'] = month_evap['month'].map(month_map)
    fig9 = px.bar(month_evap, x='month_label', y='Evaporation',
                  title='Average Evaporation per Month',
                  labels={'month_label': 'Month', 'Evaporation': 'Evaporation (mm)'},
                  color='Evaporation', color_continuous_scale='sunset')
    
    # 10. Average Evaporation per Year
    year_evap = df.groupby('year')['Evaporation'].mean().reset_index()
    fig10 = px.bar(year_evap, x='year', y='Evaporation',
                   title='Average Evaporation per Year',
                   labels={'year': 'Year', 'Evaporation': 'Evaporation (mm)'},
                   color='Evaporation', color_continuous_scale='plasma')
    
    # 11. Evaporation vs Humidity 3pm
    fig11 = px.scatter(df.dropna(subset=['Humidity3pm', 'Evaporation']), 
                      x='Humidity3pm', y='Evaporation', color='RainTomorrowLabel',
                      title='Evaporation vs Humidity 3pm',
                      color_discrete_sequence=px.colors.qualitative.D3,
                      labels={'Humidity3pm': 'Humidity 3pm (%)', 'Evaporation': 'Evaporation (mm)',
                              'RainTomorrowLabel': 'Rain Tomorrow'},
                      opacity=0.6)
    
    # 12. Evaporation vs Sunshine
    fig12 = px.scatter(df.dropna(subset=['Sunshine', 'Evaporation']), 
                      x='Sunshine', y='Evaporation', color='RainTomorrowLabel',
                      title='Evaporation vs Sunshine',
                      color_discrete_sequence=px.colors.qualitative.T10,
                      labels={'Sunshine': 'Sunshine (hours)', 'Evaporation': 'Evaporation (mm)',
                              'RainTomorrowLabel': 'Rain Tomorrow'},
                      opacity=0.6)
    
    return render_template_string(HTML_TEMPLATE,
        chart1=fig1.to_html(full_html=False),
        chart2=fig2.to_html(full_html=False),
        chart3=fig3.to_html(full_html=False),
        chart4=fig4.to_html(full_html=False),
        chart5=fig5.to_html(full_html=False),
        chart6=fig6.to_html(full_html=False),
        chart7=fig7.to_html(full_html=False),
        chart8=fig8.to_html(full_html=False),
        chart9=fig9.to_html(full_html=False),
        chart10=fig10.to_html(full_html=False),
        chart11=fig11.to_html(full_html=False),
        chart12=fig12.to_html(full_html=False)
    )

if __name__ == '__main__':
    print("Weather Dashboard: http://localhost:5000")
    app.run(debug=False, port=5000)

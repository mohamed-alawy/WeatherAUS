import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# --- Data Preparation (Same as Notebook) ---
def load_and_clean_data():
    df = pd.read_csv('weatherAUS.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Location', 'Date']).reset_index(drop=True)
    
    # Fill Temperature basics (Needed for scientific calculations)
    df['MinTemp'] = df['MinTemp'].fillna(df['MinTemp'].median())
    df['MaxTemp'] = df['MaxTemp'].fillna(df['MaxTemp'].median())
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['TempMean'] = (df['MaxTemp'] + df['MinTemp']) / 2
    
    # 1. Scientific Filling for Evaporation
    Ra = 20
    df['Evaporation_Est'] = 0.0023 * (df['TempMean'] + 17.8) * (df['TempRange'] ** 0.5) * Ra
    df['Evaporation'] = df['Evaporation'].fillna(df['Evaporation_Est'])
    
    # 2. General Null Filling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    df['RainTomorrow_Binary'] = (df['RainTomorrow'] == 'Yes').astype(int)
    return df

df_analysis = load_and_clean_data()
locations = sorted(df_analysis['Location'].unique())

def predict_rain_simple(row):
    score = 0
    if pd.notna(row['Sunshine']):
        if row['Sunshine'] < 3: score += 35
        elif row['Sunshine'] < 7: score += 15
        elif row['Sunshine'] > 11: score -= 20
    if pd.notna(row['Humidity3pm']):
        if row['Humidity3pm'] > 80: score += 30
        elif row['Humidity3pm'] > 70: score += 20
    if pd.notna(row['Cloud3pm']):
        if row['Cloud3pm'] >= 7: score += 25
        elif row['Cloud3pm'] >= 5: score += 10
    if row['RainToday'] == 'Yes': score += 20
    
    if score >= 60: return 'Very High Probability (>60%)', 'danger'
    elif score >= 40: return 'High Probability (40-60%)', 'warning'
    elif score >= 25: return 'Medium Probability (25-40%)', 'info'
    else: return 'Low Probability (<25%)', 'success'

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Weather Rain Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 20px; }
        .dashboard-container { max-width: 1200px; margin: auto; }
        .card { margin-bottom: 20px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .prediction-box { padding: 30px; border-radius: 15px; text-align: center; color: white; }
        .stat-value { font-size: 1.8rem; font-weight: bold; color: #0d6efd; }
        .bg-danger { background-color: #dc3545 !important; }
        .bg-warning { background-color: #ffc107 !important; color: black !important; }
        .bg-info { background-color: #0dcaf0 !important; color: black !important; }
        .bg-success { background-color: #198754 !important; }
    </style>
</head>
<body>
    <div class="container dashboard-container">
        <h1 class="text-center mb-4">Weather Prediction Dashboard</h1>
        
        <div class="card p-3">
            <form action="/" method="get" class="row g-3 align-items-center justify-content-center">
                <div class="col-auto">
                    <label class="form-label h5 mb-0">Select Your City:</label>
                </div>
                <div class="col-auto">
                    <select name="location" class="form-select form-select-lg" onchange="this.form.submit()">
                        {% for loc in locations %}
                        <option value="{{ loc }}" {% if loc == selected_location %}selected{% endif %}>{{ loc }}</option>
                        {% endfor %}
                    </select>
                </div>
            </form>
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="card p-3 text-center">
                    <h5>Sunshine Today</h5>
                    <div class="stat-value">{{ "%.1f"|format(current_stats.Sunshine) }} h</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card p-3 text-center">
                    <h5>Humidity (3pm)</h5>
                    <div class="stat-value">{{ "%.0f"|format(current_stats.Humidity3pm) }} %</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card p-3 text-center">
                    <h5>Cloud Cover</h5>
                    <div class="stat-value">{{ "%.0f"|format(current_stats.Cloud3pm) }}/8</div>
                </div>
            </div>
        </div>

        <div class="card prediction-box bg-{{ pred_color }}">
            <h3>Verdict for {{ selected_location }} tomorrow:</h3>
            <h1 class="display-4 font-weight-bold">{{ prediction }}</h1>
            <p class="mb-0">Latest update: {{ last_date }}</p>
        </div>

        <div class="row">
            <div class="col-md-6"><div class="card p-2">{{ chart1 | safe }}</div></div>
            <div class="col-md-6"><div class="card p-2">{{ chart2 | safe }}</div></div>
            <div class="col-12"><div class="card p-2">{{ chart3 | safe }}</div></div>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    selected_location = request.args.get('location', 'Sydney')
    
    # Filter data
    df_loc = df_analysis[df_analysis['Location'] == selected_location].sort_values('Date')
    current_day = df_loc.iloc[-1]
    
    # Prediction
    pred_text, pred_color = predict_rain_simple(current_day)
    
    # Chart 1: Sunshine (Density)
    fig1 = px.histogram(df_loc, x='Sunshine', color='RainTomorrow', 
                       marginal='box', barmode='overlay',
                       title=f'Historical Sunshine Distribution: {selected_location}',
                       color_discrete_map={'No': '#0d6efd', 'Yes': '#dc3545'},
                       template='simple_white')
    fig1.add_vline(x=current_day['Sunshine'], line_dash="dash", line_color="black", 
                  annotation_text="TODAY'S LEVEL", annotation_position="top left")

    # Chart 2: Humidity (Density)
    fig2 = px.histogram(df_loc, x='Humidity3pm', color='RainTomorrow',
                       marginal='box', barmode='overlay',
                       title=f'Historical Humidity Distribution: {selected_location}',
                       color_discrete_map={'No': '#0d6efd', 'Yes': '#dc3545'},
                       template='simple_white')
    fig2.add_vline(x=current_day['Humidity3pm'], line_dash="dash", line_color="black",
                  annotation_text="TODAY'S LEVEL", annotation_position="top left")

    # Chart 3: Cloud Cover vs Chance
    cloud_prob = df_loc.groupby('Cloud3pm')['RainTomorrow_Binary'].mean() * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=cloud_prob.index, y=cloud_prob.values, marker_color='lightgray', name='History'))
    
    # Highlight today's cloud
    today_cloud = int(current_day['Cloud3pm'])
    if today_cloud in cloud_prob.index:
        fig3.add_trace(go.Bar(x=[today_cloud], y=[cloud_prob.loc[today_cloud]], 
                           marker_color='#fd7e14', name='TODAY'))
        
    fig3.update_layout(title=f'Rain Probability (%) by Cloud Cover: {selected_location}',
                      xaxis_title='Cloud Cover (0-8)', yaxis_title='Chance of Rain (%)',
                      template='simple_white', showlegend=False)

    return render_template_string(HTML_TEMPLATE,
        locations=locations,
        selected_location=selected_location,
        current_stats=current_day,
        prediction=pred_text,
        pred_color=pred_color,
        last_date=current_day['Date'].strftime('%d %B %Y'),
        chart1=fig1.to_html(full_html=False),
        chart2=fig2.to_html(full_html=False),
        chart3=fig3.to_html(full_html=False)
    )

if __name__ == '__main__':
    print("Dashboard starting on http://localhost:5000")
    app.run(debug=True, port=5000)


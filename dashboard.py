import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request
from datetime import timedelta
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"

app = Flask(__name__)

def load_data():
    try:
        df = pd.read_csv('weatherAUS_cleaned.csv')
    except FileNotFoundError:
        # Fallback for dev if cleaned not found, try original
        df = pd.read_csv('weatherAUS.csv')

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    
    # Ensure Binary targets exist and are integers (0/1)
    if 'RainToday' in df.columns:
        # Standardize: Yes->1, No->0. Handle potential NaNs as 0 (safe default for plotting, though maybe risky for ML)
        df['RainToday_Binary'] = df['RainToday'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    if 'RainTomorrow' in df.columns:
        df['RainTomorrow_Binary'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    df = df.sort_values(['Location', 'Date']).reset_index(drop=True)
    return df

df = load_data()
locations = sorted(df['Location'].unique())
min_date = df['Date'].min().strftime('%Y-%m-%d')
max_date = df['Date'].max().strftime('%Y-%m-%d')

def predict_next_2_days(df_loc, current_date_row):
    # Extract current features
    # Use fillna(0) for safety
    sun = float(current_date_row['Sunshine']) if pd.notnull(current_date_row['Sunshine']) else 0.0
    hum = float(current_date_row['Humidity3pm']) if pd.notnull(current_date_row['Humidity3pm']) else 0.0
    cld = float(current_date_row['Cloud3pm']) if pd.notnull(current_date_row['Cloud3pm']) else 0.0
    rt = int(current_date_row['RainToday_Binary']) if pd.notnull(current_date_row['RainToday_Binary']) else 0
    
    # 1. Day 1 Logic
    # Find similar historical days (including cloud)
    # Using explicit .copy() to avoid SettingWithCopy warnings if we modify matches later
    mask1 = (
        (df_loc['Sunshine'].between(sun - 3, sun + 3)) &
        (df_loc['Humidity3pm'].between(hum - 15, hum + 15)) &
        (df_loc['Cloud3pm'].between(cld - 2, cld + 2)) &
        (df_loc['RainToday_Binary'] == rt)
    )
    matches1 = df_loc[mask1]
    
    # If too few matches, try a slightly broader search (wider ranges)
    if len(matches1) < 10:
        mask1_broad = (
            (df_loc['Sunshine'].between(sun - 6, sun + 6)) &
            (df_loc['Humidity3pm'].between(hum - 30, hum + 30)) &
            (df_loc['Cloud3pm'].between(cld - 3, cld + 3)) &
            (df_loc['RainToday_Binary'] == rt)
        )
        matches1 = df_loc[mask1_broad]
    
    if len(matches1) < 10:
        # Fallback to overall average
        prob1 = df_loc['RainTomorrow_Binary'].mean() * 100
        conf1 = 30 # Low confidence
    else:
        prob1 = matches1['RainTomorrow_Binary'].mean() * 100
        conf1 = min(len(matches1), 100)
        
    # 2. Day 2 Logic
    # Assume Day 1 outcome based on prob1
    day1_outcome = 1 if prob1 > 50 else 0
    
    # We use day 1 prediction as "input" for day 2 similarity
    # Since we don't have day 1 actual sunshine/humidity/cloud, we use averages of the 'matches1' for tomorrow
    if not matches1.empty:
        est_sun_tom = matches1['Sunshine'].mean()
        est_hum_tom = matches1['Humidity3pm'].mean()
        est_cld_tom = matches1['Cloud3pm'].mean()
    else:
        est_sun_tom = sun
        est_hum_tom = hum
        est_cld_tom = cld
        
    mask2 = (
        (df_loc['Sunshine'].between(est_sun_tom - 4, est_sun_tom + 4)) &
        (df_loc['Humidity3pm'].between(est_hum_tom - 20, est_hum_tom + 20)) &
        (df_loc['Cloud3pm'].between(est_cld_tom - 3, est_cld_tom + 3)) &
        (df_loc['RainToday_Binary'] == day1_outcome)
    )
    matches2 = df_loc[mask2]
    
    if len(matches2) < 10:
        prob2 = df_loc['RainTomorrow_Binary'].mean() * 100
        conf2 = 25
    else:
        prob2 = matches2['RainTomorrow_Binary'].mean() * 100
        conf2 = min(len(matches2), 100)
        
    return {
        'day1': {'prob': prob1, 'conf': conf1},
        'day2': {'prob': prob2, 'conf': conf2}
    }

def create_charts(df_loc, current_day=None):
    # Use nice hex colors for the dashboard
    COLOR_SUN = '#F59E0B' # Amber
    COLOR_HUM = '#3B82F6' # Blue
    COLOR_CLD = '#64748B' # Slate
    COLOR_YES = '#EF4444' # Red
    COLOR_NO = '#10B981'  # Green
    COLOR_MONTH = '#8B5CF6' # Violet

    # 1. Sunshine vs Rain Prob (Line Chart / Area)
    # Strategy: Round sunshine to nearest integer to group data, then plot trend
    # CRITICAL FIX: Drop NaNs. Do NOT fill with 0. 0 means overcast, NaN means unknown.
    df_sun = df_loc.dropna(subset=['Sunshine']).copy()
    
    if not df_sun.empty:
        df_sun['Sun_Round'] = df_sun['Sunshine'].round(0)
        # Group and calculate mean
        sun_groups = df_sun.groupby('Sun_Round')['RainTomorrow_Binary'].mean() * 100
        
        # Reindex to ensure we have a continuous x-axis (0 to max sun)
        max_sun = int(df_sun['Sunshine'].max()) if not df_sun.empty else 14
        idx_sun = np.arange(0, max_sun + 1)
        sun_groups = sun_groups.reindex(idx_sun)
        
        # Interpolate to fill gaps for a smooth line
        sun_groups_interp = sun_groups.interpolate(method='linear')
    else:
        sun_groups_interp = pd.Series([], dtype=float)

    fig1 = px.area(
        x=sun_groups_interp.index, y=sun_groups_interp.values,
        title='Sunshine vs Rain Probability',
        labels={'x': 'Sunshine (Hours)', 'y': 'Rain Probability (%)'},
    )
    fig1.update_traces(line_color=COLOR_SUN, fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)')
    fig1.update_layout(
        yaxis=dict(range=[0,105], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#e2e8f0', family='Outfit'),
        margin=dict(t=40, b=30, l=40, r=20)
    )

    # Add marker for current day
    if current_day is not None and pd.notnull(current_day['Sunshine']):
        val = round(current_day['Sunshine'])
        if val in sun_groups_interp.index:
            y_val = sun_groups_interp.loc[val]
            if pd.notnull(y_val):
                fig1.add_scatter(x=[val], y=[y_val], mode='markers', marker=dict(color='white', size=10, line=dict(color=COLOR_SUN, width=2)), showlegend=False, name='Current Day')

    # 2. Humidity vs Rain Prob (Line Chart / Area)
    # Round to nearest 2% for granularity
    df_hum = df_loc.dropna(subset=['Humidity3pm']).copy()
    
    if not df_hum.empty:
        df_hum['Hum_Round'] = (df_hum['Humidity3pm'] / 2).round() * 2
        hum_groups = df_hum.groupby('Hum_Round')['RainTomorrow_Binary'].mean() * 100
        
        # Reindex 0-100
        idx_hum = np.arange(0, 102, 2)
        hum_groups = hum_groups.reindex(idx_hum)
        hum_groups_interp = hum_groups.interpolate(method='linear')
    else:
        hum_groups_interp = pd.Series([], dtype=float)
        
    fig2 = px.area(
        x=hum_groups_interp.index, y=hum_groups_interp.values,
        title='Humidity (3pm) vs Rain Probability',
        labels={'x': 'Humidity (%)', 'y': 'Rain Probability (%)'},
    )
    fig2.update_traces(line_color=COLOR_HUM, fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)')
    fig2.update_layout(
        yaxis=dict(range=[0,105], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#e2e8f0', family='Outfit'),
        margin=dict(t=40, b=30, l=40, r=20)
    )
    
    # Add marker for current day
    if current_day is not None and pd.notnull(current_day['Humidity3pm']):
        val = 2 * round(current_day['Humidity3pm']/2)
        if val in hum_groups_interp.index:
            y_val = hum_groups_interp.loc[val]
            if pd.notnull(y_val):
                fig2.add_scatter(x=[val], y=[y_val], mode='markers', marker=dict(color='white', size=10, line=dict(color=COLOR_HUM, width=2)), showlegend=False, name='Current Day')

    # 3. Cloud Cover (Bar - Discrete)
    # Just clean it up
    cld_groups = df_loc.groupby('Cloud3pm')['RainTomorrow_Binary'].mean() * 100
    cld_groups = cld_groups.reindex(np.arange(0,9), fill_value=0)
    
    fig3 = px.bar(
        x=cld_groups.index, y=cld_groups.values,
        title='Cloud Cover (3pm) vs Rain',
        labels={'x': 'Cloud Cover (Okta)', 'y': 'Rain Probability (%)'},
        text=cld_groups.values
    )
    fig3.update_traces(marker_color=COLOR_CLD, texttemplate='%{text:.0f}%', textposition='outside')
    fig3.update_layout(
        yaxis=dict(range=[0,115], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(tickmode='array', tickvals=list(range(9))),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#e2e8f0', family='Outfit'),
        margin=dict(t=40, b=30, l=40, r=20)
    )
    
    if current_day is not None and pd.notnull(current_day['Cloud3pm']):
        val = int(current_day['Cloud3pm'])
        if 0 <= val <= 8:
             fig3.add_shape(type="rect", x0=val-0.45, x1=val+0.45, y0=0, y1=100, 
                           line=dict(color="#FDB813", width=2, dash="dot"), fillcolor="rgba(0,0,0,0)")

    # 4. Persistence (RainToday vs RainTomorrow)
    rain_perm = df_loc.groupby('RainToday')['RainTomorrow_Binary'].mean() * 100
    x_perm = ['No', 'Yes']
    y_perm = [rain_perm.get('No', 0), rain_perm.get('Yes', 0)]
    
    fig4 = px.bar(
        x=x_perm, y=y_perm,
        title='Rain Persistence',
        labels={'x': 'Rain Today?', 'y': 'Probability Prediction'},
        text=[f"{v:.1f}%" for v in y_perm],
        color=x_perm,
        color_discrete_map={'No': COLOR_NO, 'Yes': COLOR_YES}
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(
         yaxis=dict(range=[0,115], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
         showlegend=False,
         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
         font=dict(color='#e2e8f0', family='Outfit'),
         margin=dict(t=40, b=30, l=40, r=20)
    )

    # 5. Monthly Seasonality (Line Chart)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly = df_loc.groupby('Month')['RainTomorrow_Binary'].mean().reindex(month_order) * 100
    
    # Use spline for smooth monthly transition curve
    fig5 = px.line(
        x=month_order, y=monthly.values,
        title='Monthly Rain Seasonality',
        labels={'x': 'Month', 'y': 'Rain Probability (%)'},
        markers=True
    )
    fig5.update_traces(
        line_color=COLOR_MONTH, 
        line_shape='spline', 
        line_width=4,
        marker_size=8,
        marker_color='white',
        marker_line_width=2,
        marker_line_color=COLOR_MONTH
    )
    # Add an area fill
    fig5.add_trace(px.area(x=month_order, y=monthly.values).data[0])
    fig5.data[1].update(line=dict(width=0), fillcolor='rgba(139, 92, 246, 0.1)', hoverinfo='skip', showlegend=False)
    
    fig5.update_layout(
        yaxis=dict(range=[0,105], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#e2e8f0', family='Outfit'),
        margin=dict(t=40, b=80, l=40, r=40)
    )
    
    if current_day is not None:
         curr_m = current_day['Month']
         if curr_m in month_order:
             val = monthly[curr_m]
             fig5.add_scatter(x=[curr_m], y=[val], mode='markers', marker=dict(color='white', size=14, line=dict(color=COLOR_MONTH, width=3)), showlegend=False)

    return [fig1.to_html(full_html=False, config={'displayModeBar': False}),
            fig2.to_html(full_html=False, config={'displayModeBar': False}),
            fig3.to_html(full_html=False, config={'displayModeBar': False}),
            fig4.to_html(full_html=False, config={'displayModeBar': False}),
            fig5.to_html(full_html=False, config={'displayModeBar': False})]

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced WeatherAI</title>
    <!-- Flatpickr (Premium Date Picker) -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flatpickr/dist/themes/dark.css">
    <!-- Bootstrap 5 -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: #1e293b;
            --accent: #3b82f6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --danger: #ef4444;
        }
        
        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.15) 0px, transparent 50%);
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
        }
        
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.1);
        }

        .header-title {
            font-weight: 800;
            background: linear-gradient(45deg, #3b82f6, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }

        /* Controls */
        .form-select, .form-control, .flatpickr-input {
            background-color: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: white !important;
            border-radius: 12px;
            padding: 12px;
        }
        
        .form-select:focus, .form-control:focus {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border-color: var(--accent) !important;
            color: white;
            box-shadow: 0 0 0 0.2rem rgba(59, 130, 246, 0.25);
        }

        /* FIX: Flatpickr Wrapper within Bootstrap Input Group */
        .flatpickr-wrapper {
            display: block !important;
            flex: 1;
            width: 100%;
        }

        .btn-predict {
            background: linear-gradient(45deg, #3b82f6, #2563eb);
            border: none;
            border-radius: 12px;
            padding: 12px 30px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.5);
            color: white;
        }

        /* Stats & Prediction */
        .stat-item {
            text-align: center;
            padding: 15px;
            height: 100%;
        }
        
        .stat-icon {
            font-size: 1.5rem;
            color: var(--accent);
            margin-bottom: 10px;
        }
        
        .stat-val {
            font-size: 1.25rem;
            font-weight: 700;
        }

        .pred-card {
            text-align: center;
            padding: 20px;
            position: relative;
            overflow: hidden;
            height: 100%;
        }
        
        .pred-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 4px;
        }
        
        .pred-day1::before { background: var(--accent); }
        .pred-day2::before { background: #8b5cf6; }

        .prob-value {
            font-size: 3.5rem;
            font-weight: 800;
            margin: 10px 0;
            line-height: 1;
        }

        .confidence-badge {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 5px 12px;
            border-radius: 20px;
            background: rgba(255,255,255,0.1);
        }

        /* Actual Result Verification */
        .result-box {
            margin-top: 15px;
            padding: 10px;
            border-radius: 12px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .result-correct { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .result-wrong { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .result-future { background: rgba(255, 255, 255, 0.05); color: var(--text-muted); border: 1px dashed rgba(255, 255, 255, 0.1); }
        
        /* New Layout for Charts */
        .chart-container {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
        }

    </style>
</head>
<body>

<div class="container py-5">
    <!-- Header -->
    <div class="text-center mb-5">
        <h1 class="display-4 header-title mb-2">WeatherAI Analytics</h1>
    </div>

    <!-- Controls -->
    <div class="glass-card p-4 mb-5">
        <form action="/" method="get">
            <div class="row align-items-end g-3">
                <div class="col-lg-5 col-md-12 col-12">
                    <label class="form-label text-muted small text-uppercase fw-bold">Select Location</label>
                    <div class="input-group">
                        <span class="input-group-text bg-transparent border-end-0 text-white border-secondary"><i class="bi bi-geo-alt"></i></span>
                        <select name="location" class="form-select border-start-0 ps-0">
                            {% for loc in locations %}
                            <option value="{{ loc }}" {% if loc == selected_ctx.location %}selected{% endif %}>{{ loc }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="col-lg-5 col-md-12 col-12">
                    <label class="form-label text-muted small text-uppercase fw-bold">Select Date</label>
                    <div class="input-group flex-nowrap">
                        <span class="input-group-text bg-transparent border-end-0 text-white border-secondary"><i class="bi bi-calendar-event"></i></span>
                        <!-- Flatpickr Input (Type Text) -->
                        <input type="text" name="date" id="datePicker" class="form-control border-start-0 ps-0" placeholder="Select Date...">
                    </div>
                </div>
                <div class="col-lg-2 col-md-12 col-12">
                    <button type="submit" class="btn btn-predict w-100">
                        Analyze <i class="bi bi-arrow-right ms-2"></i>
                    </button>
                </div>
            </div>
        </form>
    </div>

    <!-- Current State -->
    <div class="row g-4 mb-5 row-cols-1 row-cols-md-2 row-cols-lg-4">
        <div class="col">
            <div class="glass-card stat-item">
                <div class="stat-icon"><i class="bi bi-sun"></i></div>
                <div class="text-muted small">Sunshine</div>
                <div class="stat-val">{{ "%.1f"|format(data.Sunshine) }} hrs</div>
            </div>
        </div>
        <div class="col">
            <div class="glass-card stat-item">
                <div class="stat-icon"><i class="bi bi-droplet"></i></div>
                <div class="text-muted small">Humidity (3pm)</div>
                <div class="stat-val">{{ "%.0f"|format(data.Humidity3pm) }}%</div>
            </div>
        </div>
        <div class="col">
            <div class="glass-card stat-item">
                <div class="stat-icon"><i class="bi bi-clouds"></i></div>
                <div class="text-muted small">Cloud Cover</div>
                <div class="stat-val">{{ "%.0f"|format(data.Cloud3pm) }}/8</div>
            </div>
        </div>
        <div class="col">
            <div class="glass-card stat-item">
                <div class="stat-icon"><i class="bi bi-umbrella"></i></div>
                <div class="text-muted small">Rain Today</div>
                <div class="stat-val {% if data.RainToday == 'Yes' %}text-danger{% else %}text-success{% endif %}">
                    {{ data.RainToday }}
                </div>
            </div>
        </div>
    </div>

    <!-- Prediction Cards -->
    <div class="row g-4 mb-5">
        <!-- Day 1 -->
        <div class="col-md-6 col-12">
            <div class="glass-card pred-card pred-day1">
                <h5 class="text-primary mb-1">Tomorrow</h5>
                <div class="text-muted small">{{ display.day1_date }}</div>
                
                {% if preds.day1.prob > 50 %}
                <div class="prob-value text-danger">{{ "%.0f"|format(preds.day1.prob) }}%</div>
                {% else %}
                <div class="prob-value text-success">{{ "%.0f"|format(preds.day1.prob) }}%</div>
                {% endif %}
                
                <div class="text-muted mb-3">Chance of Rain</div>
                
                {% if display.day1_res %}
                <div class="result-box {{ display.day1_cls }}">
                    <i class="bi {{ display.day1_icon }}"></i>
                    <span>Actual: <strong>{{ display.day1_val }}</strong></span>
                </div>
                {% else %}
                <div class="result-box result-future">
                    <i class="bi bi-hourglass"></i> Future Prediction
                </div>
                {% endif %}
            </div>
        </div>

        <!-- Day 2 -->
        <div class="col-md-6 col-12">
            <div class="glass-card pred-card pred-day2">
                <h5 style="color: #8b5cf6;" class="mb-1">Day After Tomorrow</h5>
                <div class="text-muted small">{{ display.day2_date }}</div>
                
                {% if preds.day2.prob > 50 %}
                <div class="prob-value text-danger">{{ "%.0f"|format(preds.day2.prob) }}%</div>
                {% else %}
                <div class="prob-value text-success">{{ "%.0f"|format(preds.day2.prob) }}%</div>
                {% endif %}

                <div class="text-muted mb-3">Chance of Rain</div>
                
                {% if display.day2_res %}
                <div class="result-box {{ display.day2_cls }}">
                    <i class="bi {{ display.day2_icon }}"></i>
                    <span>Actual: <strong>{{ display.day2_val }}</strong></span>
                </div>
                {% else %}
                <div class="result-box result-future">
                    <i class="bi bi-hourglass"></i> Future Prediction
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Charts Grid -->
    <h4 class="mb-4 ps-2 border-start border-4 border-primary">Deep Analysis</h4>
    <div class="row g-4 row-cols-1 row-cols-lg-2">
        <div class="col">
            <div class="glass-card p-2 h-100 chart-container">
                {{ charts[0] | safe }}
            </div>
        </div>
        <div class="col">
            <div class="glass-card p-2 h-100 chart-container">
                {{ charts[1] | safe }}
            </div>
        </div>
        <div class="col">
            <div class="glass-card p-2 h-100 chart-container">
                {{ charts[2] | safe }}
            </div>
        </div>
        <div class="col">
            <div class="glass-card p-2 h-100 chart-container">
                {{ charts[3] | safe }}
            </div>
        </div>
    </div>

    <!-- Full-width Monthly Chart -->
    <div class="row mt-4">
        <div class="col-12">
            <div class="glass-card p-3 h-100 chart-container">
                {{ charts[4] | safe }}
            </div>
        </div>
    </div>

    <div class="text-center mt-5 text-muted small">
        <p>&copy; 2026 WeatherAI Pro. Powered by Flask & Plotly.</p>
    </div>
</div>

<!-- Scripts -->
<script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
<script>
    flatpickr("#datePicker", {
        defaultDate: "{{ selected_ctx.date }}",
        minDate: "{{ selected_ctx.min_date }}",
        maxDate: "{{ selected_ctx.max_date }}",
        dateFormat: "Y-m-d",
        altInput: true,
        altFormat: "F j, Y",
        locale: "en",
        theme: "dark"
    });
</script>

</body>
</html>
'''

@app.route('/')
def index():
    # 1. Selection Logic
    sel_loc = request.args.get('location', 'Sydney')
    sel_date_str = request.args.get('date', max_date)
    sel_date = pd.to_datetime(sel_date_str)
    
    # 2. Filter & Context
    df_loc = df[df['Location'] == sel_loc].copy()
    
    # Get current day row
    try:
        # Try exact match, else fallback to closest past date
        current_idx = df_loc[df_loc['Date'] <= sel_date].last_valid_index()
        if current_idx is None:
            current_day = df_loc.iloc[0]
            sel_date = current_day['Date']
        else:
            current_day = df_loc.loc[current_idx]
    except Exception:
        current_day = df_loc.iloc[-1]

    # 3. Predict
    # Use helper to get probabilities
    preds = predict_next_2_days(df_loc, current_day)
    
    # 4. Generate Charts (highlight selected day's values)
    charts_html = create_charts(df_loc, current_day)
    
    # 5. Verification Logic
    d1_date = sel_date + timedelta(days=1)
    d2_date = sel_date + timedelta(days=2)
    
    display_info = {
        'day1_date': d1_date.strftime('%B %d, %Y'),
        'day2_date': d2_date.strftime('%B %d, %Y'),
        'day1_res': False, 'day2_res': False
    }
    
    # Check Day 1 Actual
    row_d1 = df_loc[df_loc['Date'] == d1_date]
    if not row_d1.empty:
        actual = row_d1.iloc[0]['RainToday'] # Rain for that day
        
        display_info['day1_res'] = True
        display_info['day1_val'] = actual
        pred_bool = preds['day1']['prob'] >= 50
        act_bool = actual == 'Yes'
        display_info['day1_cls'] = 'result-correct' if pred_bool == act_bool else 'result-wrong'
        display_info['day1_icon'] = 'bi-check-circle-fill' if pred_bool == act_bool else 'bi-x-circle-fill'

    # Check Day 2 Actual
    row_d2 = df_loc[df_loc['Date'] == d2_date]
    if not row_d2.empty:
        actual = row_d2.iloc[0]['RainToday']
        display_info['day2_res'] = True
        display_info['day2_val'] = actual
        pred_bool = preds['day2']['prob'] >= 50
        act_bool = actual == 'Yes'
        display_info['day2_cls'] = 'result-correct' if pred_bool == act_bool else 'result-wrong'
        display_info['day2_icon'] = 'bi-check-circle-fill' if pred_bool == act_bool else 'bi-x-circle-fill'

    context = {
        'location': sel_loc,
        'date': sel_date.strftime('%Y-%m-%d'),
        'min_date': min_date,
        'max_date': max_date
    }

    return render_template_string(HTML_TEMPLATE,
        locations=locations,
        selected_ctx=context,
        data=current_day,
        preds=preds,
        display=display_info,
        charts=charts_html
    )

if __name__ == '__main__':
    print("Starting Premium Dashboard...")
    app.run(debug=True, port=5000)

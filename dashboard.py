import pandas as pd
from flask import Flask, render_template_string, request
from datetime import timedelta
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"

app = Flask(__name__)

def load_data():
    df = pd.read_csv('weatherAUS_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    df = df.sort_values(['Location', 'Date']).reset_index(drop=True)
    return df

df = load_data()
locations = sorted(df['Location'].unique())
min_date = df['Date'].min().strftime('%Y-%m-%d')
max_date = df['Date'].max().strftime('%Y-%m-%d')

def predict_next_2_days(df_loc, current_date_row):
    # Extract current features
    sun = current_date_row['Sunshine']
    hum = current_date_row['Humidity3pm']
    cld = current_date_row['Cloud3pm']
    rt = 1 if current_date_row['RainToday'] == 'Yes' else 0
    
    # 1. Day 1 Logic
    # Find similar historical days (including cloud)
    mask1 = (
        (df_loc['Sunshine'].between(sun - 3, sun + 3)) &
        (df_loc['Humidity3pm'].between(hum - 15, hum + 15)) &
        (df_loc['Cloud3pm'].between(cld - 1, cld + 1)) &
        (df_loc['RainToday_Binary'] == rt)
    )
    matches1 = df_loc[mask1]
    
    # If too few matches, try a slightly broader search (wider ranges)
    if len(matches1) < 10:
        mask1_broad = (
            (df_loc['Sunshine'].between(sun - 6, sun + 6)) &
            (df_loc['Humidity3pm'].between(hum - 30, hum + 30)) &
            (df_loc['Cloud3pm'].between(cld - 2, cld + 2)) &
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
        (df_loc['Cloud3pm'].between(est_cld_tom - 2, est_cld_tom + 2)) &
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

def create_charts(df_loc):
    # 1. Cloud Cover vs Rain Probability (Bar)
    cloud_gb = df_loc.groupby('Cloud3pm')['RainTomorrow_Binary'].mean().reset_index()
    cloud_gb['RainProb'] = cloud_gb['RainTomorrow_Binary'] * 100
    
    fig1 = px.bar(
        cloud_gb, x='Cloud3pm', y='RainProb',
        title=f"Cloud Cover Impact",
        labels={'Cloud3pm': 'Cloud Cover (0-8)', 'RainProb': 'Rain Probability (%)'},
        color='RainProb', color_continuous_scale='RdBu_r'
    )
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")

    # 2. Humidity Distribution (Violin/Box)
    fig2 = px.box(
        df_loc, x='RainTomorrow', y='Humidity3pm',
        color='RainTomorrow',
        title="Humidity Dist. by Rain Outcome",
        labels={'Humidity3pm': 'Humidity 3pm (%)'},
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")

    # 3. Sunshine vs Rain (Histogram)
    fig3 = px.histogram(
        df_loc, x='Sunshine', color='RainTomorrow',
        barmode='overlay', opacity=0.75,
        title="Sunshine Hours Distribution",
        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
    )
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")

    # 4. Seasonality (Line)
    # Ensure correct month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly = df_loc.groupby('Month')['RainTomorrow_Binary'].mean().reindex(month_order).reset_index()
    monthly['RainProb'] = monthly['RainTomorrow_Binary'] * 100
    
    fig4 = px.line(
        monthly, x='Month', y='RainProb',
        title="Monthly Rain Seasonality",
        markers=True,
        line_shape='spline'
    )
    fig4.update_traces(line_color='#FFE66D', line_width=4)
    fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")

    return [fig1.to_html(full_html=False, config={'displayModeBar': False}),
            fig2.to_html(full_html=False, config={'displayModeBar': False}),
            fig3.to_html(full_html=False, config={'displayModeBar': False}),
            fig4.to_html(full_html=False, config={'displayModeBar': False})]

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
                
                <div class="prob-value {% if preds.day1.prob > 50 %}text-danger{% else %}text-success{% endif %}">
                    {{ "%.0f"|format(preds.day1.prob) }}%
                </div>
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
                
                <div class="prob-value {% if preds.day2.prob > 50 %}text-danger{% else %}text-success{% endif %}">
                    {{ "%.0f"|format(preds.day2.prob) }}%
                </div>
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
            <div class="glass-card p-2 h-100">
                {{ charts[0] | safe }}
            </div>
        </div>
        <div class="col">
            <div class="glass-card p-2 h-100">
                {{ charts[1] | safe }}
            </div>
        </div>
        <div class="col">
            <div class="glass-card p-2 h-100">
                {{ charts[2] | safe }}
            </div>
        </div>
        <div class="col">
            <div class="glass-card p-2 h-100">
                {{ charts[3] | safe }}
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
    preds = predict_next_2_days(df_loc, current_day)
    
    # 4. Generate Charts
    charts_html = create_charts(df_loc)
    
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
        # NOTE: Dataset RainTomorrow is for next day. 
        # But for verification, if we predicted for d1, we check d1's RainToday or previous day's RainTomorrow?
        # Simpler: Check d1's 'RainToday' value which tells if it rained on d1.
        
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

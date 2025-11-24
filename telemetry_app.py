import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Rockoon Telemetry", layout="wide")
st.title("Rockoon Telemetry Pro")

# --- 1. データ処理 ---
@st.cache_data
def load_and_process_data(apply_filter=True):
    file_path = 'flight_data.csv'
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='shift_jis')

    df.columns = [c.strip().lower() for c in df.columns]
    
    col_map = {}
    for c in df.columns:
        if 'time' in c: col_map[c] = 'timestamp'
        if 'lat' in c: col_map[c] = 'lat'
        if 'lon' in c: col_map[c] = 'lon'
        if 'alt' in c or 'height' in c: col_map[c] = 'altitude'
        if 'ss' in c or 'rssi' in c: col_map[c] = 'rssi'
    
    df = df.rename(columns=col_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    if 'altitude' not in df.columns: df['altitude'] = 0

    if apply_filter:
        df = df[(df['lat'] > 20) & (df['lat'] < 50)]
        df = df[(df['lon'] > 120) & (df['lon'] < 150)]
        w = 5
        for col in ['lat', 'lon', 'altitude']:
            df[col] = df[col].rolling(window=w, center=True, min_periods=1).median()

    # --- 【修正】速度計算ロジック ---
    # 「1つ前」ではなく「5つ前」のデータと比較する設定
    # これにより、細かいデータ重複による「速度0」を防ぎます
    calc_span = 5  

    # 5つ前のデータとの時間差 (秒)
    df['dt'] = df['timestamp'].diff(periods=calc_span).dt.total_seconds()
    df['dt'] = df['dt'].replace(0, np.nan) 

    # 1. 垂直速度 (5つ前との高度差 / 時間差)
    df['d_alt'] = df['altitude'].diff(periods=calc_span)
    df['climb_rate'] = df['d_alt'] / df['dt']
    df['climb_rate'] = df['climb_rate'].replace([np.inf, -np.inf], np.nan)
    
    # さらに移動平均で滑らかにする
    df['climb_rate'] = df['climb_rate'].rolling(5, center=True, min_periods=1).mean()

    # 2. 水平速度 (5つ前との距離差 / 時間差)
    mean_lat = np.radians(df['lat'].mean())
    
    # 5つ前との緯度・経度差
    dy = df['lat'].diff(periods=calc_span) * 111000
    dx = df['lon'].diff(periods=calc_span) * 111000 * np.cos(mean_lat)
    
    dist = np.sqrt(dy**2 + dx**2)
    
    df['ground_speed'] = dist / df['dt']
    df['ground_speed'] = df['ground_speed'].replace([np.inf, -np.inf], np.nan)
    df['ground_speed'] = df['ground_speed'].rolling(5, center=True, min_periods=1).mean()

    # アンテナ追尾 (ここは変わりません)
    if len(df) > 0:
        h_lat = df['lat'].iloc[0]
        h_lon = df['lon'].iloc[0]
        h_alt = df['altitude'].iloc[0]

        dy_km = (df['lat'] - h_lat) * 111.0
        lon_factor = 111.0 * np.cos(np.radians(h_lat))
        dx_km = (df['lon'] - h_lon) * lon_factor
        
        dist_sq = dx_km**2 + dy_km**2
        df['ground_dist_km'] = np.sqrt(dist_sq)
        df['rel_alt_km'] = (df['altitude'] - h_alt) / 1000.0

        az = np.degrees(np.arctan2(dx_km, dy_km))
        df['azimuth'] = (az + 360) % 360
        el = np.degrees(np.arctan2(df['rel_alt_km'], df['ground_dist_km']))
        df['elevation'] = el
    
    return df

# --- サイドバー ---
with st.sidebar:
    st.header("Control Panel")
    use_filter = st.checkbox("Noise Filter", value=True)
    map_style = st.selectbox(
        "Map Style", ["open-street-map", "carto-positron"], index=0
    )

# --- 実行 ---
try:
    df = load_and_process_data(use_filter)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

if len(df) == 0:
    if use_filter:
        st.warning("Data is 0. Turning off filter.")
        df = load_and_process_data(False)
    else:
        st.error("No Data.")
        st.stop()

# スライダー
slider_idx = st.slider("Time Scrubbing", 0, len(df) - 1, len(df) - 1, 1)
cur = df.iloc[slider_idx]
hist = df.iloc[:slider_idx+1]
gmap_url = f"https://www.google.com/maps/search/?api=1&query={cur['lat']},{cur['lon']}"

# --- ステータス判定 ---
max_alt_idx = df['altitude'].argmax()
start_alt = df['altitude'].iloc[0]
current_alt = cur['altitude']

if slider_idx <= max_alt_idx:
    if (current_alt - start_alt) < 50:
        status_text = "STANDBY"
        status_color = "#607D8B"
    else:
        status_text = "ASCENT"
        status_color = "#FF5252"
else:
    if (current_alt - start_alt) < 100:
        status_text = "SPLASHDOWN"
        status_color = "#00BCD4"
    else:
        status_text = "DESCENT"
        status_color = "#448AFF"

# --- レイアウト ---
col1, col2, col3 = st.columns([2.8, 1.7, 2.5])

# 1. 地図
with col1:
    st.subheader("Map")
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=hist['lat'], lon=hist['lon'], mode='lines',
        line=dict(width=4, color='cyan'), name='Path'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[cur['lat']], lon=[cur['lon']], mode='markers',
        marker=dict(size=15, color='orange'),
        text=f"{cur['altitude']:.0f}m", name='Pos'
    ))
    fig.update_layout(
        mapbox_style=map_style,
        mapbox=dict(center=dict(lat=cur['lat'], lon=cur['lon']), zoom=10),
        height=500, margin=dict(l=0,r=0,t=0,b=0), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"[📍 **Google Maps**]({gmap_url})")

# 2. 情報
with col2:
    st.markdown(f"""
    <div style="background-color:{status_color};padding:10px;border-radius:5px;
    text-align:center;margin-bottom:20px;color:white;font-weight:bold;
    font-size:24px;letter-spacing:2px;">{status_text}</div>
    """, unsafe_allow_html=True)

    st.subheader("Tracking")
    st.metric("Azimuth", f"{cur['azimuth']:.1f}°", "North=0°")
    st.metric("Elevation", f"{cur['elevation']:.1f}°", "Up=+")
    st.metric("Distance", f"{cur['ground_dist_km']:.2f} km")
    st.divider()
    st.metric("Altitude", f"{cur['altitude']:.0f} m")
    
    # NaNなら0.0を表示
    v_spd = cur['climb_rate'] if not np.isnan(cur['climb_rate']) else 0.0
    h_spd = cur['ground_speed'] if not np.isnan(cur['ground_speed']) else 0.0
    
    st.metric("Vert Spd", f"{v_spd:.1f} m/s")
    st.metric("Gnd Spd", f"{h_spd:.1f} m/s")

# 3. グラフ
with col3:
    st.subheader("Logs")
    fig_alt = px.line(hist, x='timestamp', y='altitude', title="Altitude")
    fig_alt.update_traces(line_color='#4CAF50')
    fig_alt.update_layout(height=200, margin=dict(t=30,b=0,l=0,r=0))
    st.plotly_chart(fig_alt, use_container_width=True)

    fig_spd = go.Figure()
    fig_spd.add_trace(go.Scatter(
        x=hist['timestamp'], y=hist['climb_rate'],
        name='Vertical', line=dict(color='#FF5722')
    ))
    fig_spd.add_trace(go.Scatter(
        x=hist['timestamp'], y=hist['ground_speed'],
        name='Ground', line=dict(color='#9C27B0')
    ))
    fig_spd.update_layout(
        title="Speed (m/s)", height=200,
        margin=dict(t=30,b=0,l=0,r=0), showlegend=True
    )
    st.plotly_chart(fig_spd, use_container_width=True)

# 詳細
st.divider()
st.subheader("Data Details")
cols = ['timestamp', 'lat', 'lon', 'altitude', 'rssi', 
        'climb_rate', 'ground_speed', 'azimuth', 
        'elevation', 'ground_dist_km']
use_cols = [c for c in cols if c in df.columns]
show_df = df[use_cols].copy().round(2)

csv = show_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "data.csv", "text/csv")
st.dataframe(show_df, use_container_width=True)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import time
import textwrap

# ワイドレイアウト・絵文字なしタイトル
st.set_page_config(page_title="Rockoon Telemetry", layout="wide")
st.title("Rockoon Telemetry System")

# --- 定数 ---
SEQUENCE_MAP = {0: "STANDBY", 1: "LAUNCH", 2: "THRUSTER", 3: "IGNITION", 4: "BURST", 5: "DESCENT", 6: "SPLASHDOWN"}

# --- タイムライン描画 (準備エリア復活・色固定版) ---
def render_timeline(current_alt, is_descent, phase_label):
    # === 設定値 ===
    BURST_ALT = 30000
    
    # バー上の位置設定 (%)
    POS_START = 10  # LAUNCH地点 (ここより左は準備エリア)
    POS_BURST = 90  # 破裂地点
    
    # マーカー位置計算 (POS_START ~ POS_BURST の間に配置)
    def get_pos(alt):
        ratio = alt / BURST_ALT
        return POS_START + (ratio * (POS_BURST - POS_START))

    pct_lnc = POS_START
    pct_ign = get_pos(15000)
    pct_thr = get_pos(20000)
    pct_brs = POS_BURST

    # === 進捗率計算 ===
    if phase_label == "STANDBY":
        # 準備中: 0% ~ 10% の真ん中
        progress = 5
    elif not is_descent:
        # 上昇中: 10% -> 90%
        progress = get_pos(current_alt)
        progress = min(POS_BURST, max(POS_START, progress))
    else:
        # 降下中: 90% -> 100%
        descent_ratio = 1.0 - (current_alt / BURST_ALT)
        if descent_ratio < 0: descent_ratio = 0
        if descent_ratio > 1: descent_ratio = 1
        progress = POS_BURST + (descent_ratio * (100 - POS_BURST))

    # 色設定: 常に青 (SpaceX Blue)
    bar_color = "#00A3E0"

    # アクティブ判定
    c_lnc = "active" if current_alt >= 10 or is_descent else ""
    c_ign = "active" if current_alt >= 15000 or is_descent else ""
    c_thr = "active" if current_alt >= 20000 or is_descent else ""
    c_brs = "active" if current_alt >= 30000 or is_descent else ""

    # HTML生成
    html_code = textwrap.dedent(f"""
        <style>
            body {{ margin: 0; padding: 0; background-color: #0E1117; font-family: sans-serif; }}
            .timeline-wrapper {{
                background-color: #0E1117;
                padding: 5px 0 35px 0;
                width: 100%;
                box-sizing: border-box;
            }}
            .header-flex {{
                display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 5px;
            }}
            .timeline-container {{
                position: relative;
                width: 100%;
                height: 40px;
                margin-top: 15px;
            }}
            /* ベースライン */
            .base-line {{
                position: absolute; top: 50%; left: 0; width: 100%; height: 2px;
                background-color: #333; transform: translateY(-50%); z-index: 0;
            }}
            /* 左端の点線（準備エリア） */
            .prep-line {{
                position: absolute; top: 50%; left: 0; width: {POS_START}%; height: 2px;
                background: repeating-linear-gradient(90deg, #333, #333 4px, transparent 4px, transparent 8px);
                transform: translateY(-50%); z-index: 0;
            }}
            .progress-bar {{
                position: absolute; top: 50%; left: 0; height: 2px;
                background-color: {bar_color}; width: {progress}%;
                transform: translateY(-50%); transition: width 0.3s ease-out; z-index: 1;
                box-shadow: 0 0 8px {bar_color};
            }}
            .marker {{
                position: absolute; top: 50%; transform: translate(-50%, -50%);
                width: 10px; height: 10px; background-color: #0E1117;
                border: 2px solid #666; border-radius: 50%; z-index: 2;
            }}
            .marker.active {{
                background-color: #fff; border-color: #fff;
                box-shadow: 0 0 8px rgba(255,255,255,0.8);
            }}
            .marker-label {{
                position: absolute; top: 30px; transform: translateX(-50%);
                color: #888; font-size: 10px; font-weight: bold;
                text-align: center; line-height: 1.1; width: 80px; white-space: nowrap;
            }}
            .phase-badge {{
                background-color: #1f1f1f; border: 1px solid #444; color: #eee;
                padding: 4px 12px; border-radius: 4px; font-weight: bold;
                font-size: 14px; letter-spacing: 1px;
            }}
        </style>
        <div class="timeline-wrapper">
            <div class="header-flex">
                <span style="color:#666; font-size:12px; letter-spacing:1px; font-weight:bold;">MISSION TIMELINE</span>
                <div class="phase-badge">PHASE: {phase_label}</div>
            </div>
            <div class="timeline-container">
                <div class="base-line"></div>
                <div class="prep-line"></div>
                <div class="progress-bar"></div>
                
                <div class="marker {c_lnc}" style="left: {pct_lnc}%;"></div>
                <div class="marker-label" style="left: {pct_lnc}%;">LAUNCH<br>0km</div>

                <div class="marker {c_ign}" style="left: {pct_ign}%;"></div>
                <div class="marker-label" style="left: {pct_ign}%;">IGNITION<br>15km</div>

                <div class="marker {c_thr}" style="left: {pct_thr}%;"></div>
                <div class="marker-label" style="left: {pct_thr}%;">THRUSTER<br>20km</div>

                <div class="marker {c_brs}" style="left: {pct_brs}%;"></div>
                <div class="marker-label" style="left: {pct_brs}%;">BURST<br>30km</div>
            </div>
        </div>
    """)
    components.html(html_code, height=110, scrolling=False)

# --- KML生成 ---
def generate_kml(df):
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Rockoon Flight Path</name>
    <Style id="yellowLineGreenPoly">
      <LineStyle><color>7f00ffff</color><width>4</width></LineStyle>
      <PolyStyle><color>7f00ff00</color></PolyStyle>
    </Style>
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#yellowLineGreenPoly</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
"""
    kml_footer = """        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
    coords = ""
    for _, row in df.iterrows():
        coords += f"{row['lon']},{row['lat']},{row['altitude']} \n"
    return kml_header + coords + kml_footer

# --- 計算処理 ---
def calculate_metrics(df):
    df['dt'] = df['timestamp'].diff(5).dt.total_seconds().replace(0, np.nan)
    df['climb_rate'] = (df['altitude'].diff(5) / df['dt']).replace([np.inf, -np.inf], np.nan)
    df['climb_rate'] = df['climb_rate'].rolling(5, center=True).mean()
    
    mean_lat = np.radians(df['lat'].mean())
    dy = df['lat'].diff(5) * 111000
    dx = df['lon'].diff(5) * 111000 * np.cos(mean_lat)
    dist = np.sqrt(dy**2 + dx**2)
    df['ground_speed'] = (dist / df['dt']).replace([np.inf, -np.inf], np.nan)
    df['ground_speed'] = df['ground_speed'].rolling(5, center=True).mean()

    if len(df) > 0:
        h_lat = df['lat'].iloc[0]
        h_lon = df['lon'].iloc[0]
        h_alt = df['altitude'].iloc[0]
        dy_km = (df['lat'] - h_lat) * 111.0
        dx_km = (df['lon'] - h_lon) * (111.0 * np.cos(np.radians(h_lat)))
        df['ground_dist_km'] = np.sqrt(dx_km**2 + dy_km**2)
        df['rel_alt_km'] = (df['altitude'] - h_alt) / 1000.0
        df['slant_dist_km'] = np.sqrt(df['ground_dist_km']**2 + df['rel_alt_km']**2)
        df['azimuth'] = (np.degrees(np.arctan2(dx_km, dy_km)) + 360) % 360
        df['elevation'] = np.degrees(np.arctan2(df['rel_alt_km'], df['ground_dist_km']))
    return df

# --- Manual読み込み ---
@st.cache_data
def load_manual_data(file, apply_filter):
    try:
        try: df = pd.read_csv(file, encoding='utf-8')
        except: df = pd.read_csv(file, encoding='shift_jis')
        df.columns = [c.strip().lower() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if 'time' in c: col_map[c] = 'timestamp'
            if 'lat' in c: col_map[c] = 'lat'
            if 'lon' in c: col_map[c] = 'lon'
            if 'alt' in c: col_map[c] = 'altitude'
            if 'ss' in c or 'rssi' in c: col_map[c] = 'rssi'
            if 'seq' in c or 'phase' in c: col_map[c] = 'seq'
        df = df.rename(columns=col_map)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        if 'altitude' not in df.columns: df['altitude'] = 0
        if 'seq' not in df.columns: df['seq'] = 0
        
        if apply_filter:
            df = df[(df['lat'] > 20) & (df['lat'] < 50)]
            df = df[(df['lon'] > 120) & (df['lon'] < 150)]
            w = 5
            for col in ['lat', 'lon', 'altitude']:
                df[col] = df[col].rolling(window=w, center=True, min_periods=1).median()
        return calculate_metrics(df)
    except: return pd.DataFrame()

# --- Real-time読み込み ---
def load_realtime_data(file_path):
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None, 
                         names=['date', 'time', 'altitude', 'lat', 'lon', 'rssi', 'seq'],
                         on_bad_lines='skip')
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'lat', 'lon'])
        df = df[(df['lat'] > 20) & (df['lat'] < 50)]
        return calculate_metrics(df)
    except: return pd.DataFrame()

# --- サイドバー設定 ---
with st.sidebar:
    st.header("System Mode")
    mode = st.radio("Select Mode", ["Manual Analysis", "Real-time Monitor"])
    st.markdown("---")
    if mode == "Manual Analysis":
        uploaded_file = st.file_uploader("Upload Log", type=['csv', 'txt'])
    else:
        log_path = st.text_input("Log Path", "flight_log.csv")
        start_btn = st.button("START MONITORING", type="primary")
    st.markdown("---")
    use_filter = st.checkbox("Noise Filter", value=True)
    color_mode = st.radio("Path Color", ["Altitude", "RSSI"])
    pitch = st.slider("3D Pitch", 0, 60, 45)

# --- 共通描画 ---
def render_dashboard(cur, hist, full_df):
    max_alt_idx = full_df['altitude'].argmax()
    start_alt = full_df['altitude'].iloc[0]
    is_descent = cur.name >= max_alt_idx
    v_speed = cur['climb_rate'] if not np.isnan(cur['climb_rate']) else 0.0

    if not is_descent:
        if (cur['altitude'] - start_alt) < 10:
            status_text = "STANDBY"
            status_color = "#607D8B"
        elif v_speed > 10.0:
            status_text = "LAUNCH"
            status_color = "#FFC107"
        else:
            status_text = "ASCENT"
            status_color = "#D32F2F"
        burst_disp = "---- m"
    else:
        if (cur['altitude'] - start_alt) < 100:
            status_text = "SPLASHDOWN"
            status_color = "#0288D1"
        else:
            status_text = "DESCENT"
            status_color = "#303F9F"
        burst_disp = f"{full_df.iloc[max_alt_idx]['altitude']:.0f} m"

    # タイムライン表示
    render_timeline(cur['altitude'], is_descent, status_text)

    col_btn, col_stats = st.columns([1, 3])
    with col_btn:
        st.download_button("Get KML", generate_kml(full_df), "flight.kml", use_container_width=True)
        cols = ['timestamp', 'lat', 'lon', 'altitude', 'rssi', 'climb_rate', 'ground_speed', 'seq']
        use = [c for c in cols if c in full_df.columns]
        st.download_button("Download CSV", full_df[use].to_csv(index=False).encode('utf-8'), "data.csv", use_container_width=True)

    with col_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Burst Alt", burst_disp)
        max_spd = hist['climb_rate'].max()
        c2.metric("Max Speed", f"{max_spd:.1f} m/s" if not np.isnan(max_spd) else "0.0 m/s")
        max_dist = hist['ground_dist_km'].max()
        c3.metric("Max Dist", f"{max_dist:.2f} km")
        c4.markdown(f"<div style='background:{status_color};color:white;padding:10px;text-align:center;border-radius:4px;font-weight:bold;font-size:18px;'>{status_text}</div>", unsafe_allow_html=True)

    st.markdown("---")

    col_map, col_track, col_chart = st.columns([2.7, 1.8, 2.5])

    with col_map:
        tab2d, tab3d = st.tabs(["2D Map", "3D View"])
        with tab2d:
            fig = go.Figure()
            if "RSSI" in color_mode and 'rssi' in hist.columns:
                fig.add_trace(go.Scattermapbox(lat=hist['lat'], lon=hist['lon'], mode='markers+lines', marker=dict(size=4, color=hist['rssi'], colorscale='RdYlBu', cmin=-120, cmax=-40)))
            else:
                fig.add_trace(go.Scattermapbox(lat=hist['lat'], lon=hist['lon'], mode='lines', line=dict(width=4, color='cyan')))
            
            fig.add_trace(go.Scattermapbox(lat=[cur['lat']], lon=[cur['lon']], mode='markers', marker=dict(size=12, color='orange')))
            
            if is_descent:
                bp = full_df.iloc[max_alt_idx]
                fig.add_trace(go.Scattermapbox(lat=[bp['lat']], lon=[bp['lon']], mode='markers', marker=dict(size=10, color='red')))

            fig.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=cur['lat'], lon=cur['lon']), zoom=10), height=450, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"[Google Maps](https://www.google.com/maps/search/?api=1&query={cur['lat']},{cur['lon']})")

        with tab3d:
            path_data = hist[['lon', 'lat', 'altitude']].values.tolist()
            layer_path = pdk.Layer("PathLayer", data=[{"path": path_data}], get_path="path", get_color=[255, 215, 0], width_min_pixels=3)
            layer_pos = pdk.Layer("ScatterplotLayer", data=pd.DataFrame([cur]), get_position=["lon", "lat", "altitude"], get_color=[255, 165, 0], get_radius=200, pickable=True)
            view = pdk.ViewState(latitude=cur['lat'], longitude=cur['lon'], zoom=10, pitch=pitch, bearing=0)
            st.pydeck_chart(pdk.Deck(layers=[layer_path, layer_pos], initial_view_state=view, map_style="light", tooltip={"text": "{altitude}m"}))

    with col_track:
        st.subheader("Tracking")
        c_az, c_el = st.columns(2)
        c_az.metric("Azimuth", f"{cur['azimuth']:.1f}°", "N=0°")
        c_el.metric("Elevation", f"{cur['elevation']:.1f}°", "Up=+")
        st.divider()
        st.metric("Horizontal Dist", f"{cur['ground_dist_km']:.2f} km")
        st.metric("Direct Dist", f"{cur['slant_dist_km']:.2f} km")
        st.divider()
        st.metric("Altitude", f"{cur['altitude']:.0f} m")
        v = 0.0 if np.isnan(cur['climb_rate']) else cur['climb_rate']
        h = 0.0 if np.isnan(cur['ground_speed']) else cur['ground_speed']
        st.metric("Vert Spd", f"{v:.1f} m/s")
        st.metric("Gnd Spd", f"{h:.1f} m/s")

    with col_chart:
        st.subheader("Logs")
        fig_alt = px.line(hist, x='timestamp', y='altitude', title="Altitude")
        fig_alt.update_traces(line_color='#4CAF50')
        if is_descent: fig_alt.add_hline(y=full_df.iloc[max_alt_idx]['altitude'], line_dash="dash", line_color="red")
        fig_alt.update_layout(height=200, margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_alt, use_container_width=True)

        fig_spd = go.Figure()
        fig_spd.add_trace(go.Scatter(x=hist['timestamp'], y=hist['climb_rate'], name='Vert', line=dict(color='#FF5722')))
        fig_spd.add_trace(go.Scatter(x=hist['timestamp'], y=hist['ground_speed'], name='Gnd', line=dict(color='#9C27B0')))
        fig_spd.update_layout(title="Speed", height=200, margin=dict(t=30,b=0,l=0,r=0), showlegend=True)
        st.plotly_chart(fig_spd, use_container_width=True)

        if 'rssi' in hist.columns:
            fig_rssi = px.line(hist, x='timestamp', y='rssi', title="Signal Strength")
            fig_rssi.update_traces(line_color='#2196F3')
            fig_rssi.update_layout(height=200, margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig_rssi, use_container_width=True)

# --- 実行 ---
if mode == "Manual Analysis":
    if uploaded_file: df = load_manual_data(uploaded_file, use_filter)
    else:
        try: df = load_manual_data('flight_data.csv', use_filter)
        except: st.info("Upload file."); st.stop()
    if len(df) == 0: st.error("No Data."); st.stop()
    
    slider_idx = st.slider("Time Scrubbing", 0, len(df) - 1, len(df) - 1, 1)
    render_dashboard(df.iloc[slider_idx], df.iloc[:slider_idx+1], df)
    st.divider()
    st.dataframe(df.round(2), use_container_width=True)

elif mode == "Real-time Monitor":
    ph = st.empty()
    if start_btn:
        while True:
            df_live = load_realtime_data(log_path)
            if len(df_live) > 0:
                with ph.container():
                    render_dashboard(df_live.iloc[-1], df_live, df_live)
                    st.success(f"Live Updating... Last: {df_live.iloc[-1]['timestamp']}")
            else:
                ph.warning(f"Waiting... ({log_path})")
            time.sleep(1)

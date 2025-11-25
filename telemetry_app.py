import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np

st.set_page_config(page_title="Rockoon Telemetry", layout="wide")
st.title("Rockoon Telemetry System")

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

# --- データ処理 ---
@st.cache_data
def process_data(df, apply_filter=True):
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
        h_lat, h_lon, h_alt = df['lat'].iloc[0], df['lon'].iloc[0], df['altitude'].iloc[0]
        dy_km = (df['lat'] - h_lat) * 111.0
        dx_km = (df['lon'] - h_lon) * (111.0 * np.cos(np.radians(h_lat)))
        df['ground_dist_km'] = np.sqrt(dx_km**2 + dy_km**2)
        df['rel_alt_km'] = (df['altitude'] - h_alt) / 1000.0
        df['slant_dist_km'] = np.sqrt(df['ground_dist_km']**2 + df['rel_alt_km']**2)
        df['azimuth'] = (np.degrees(np.arctan2(dx_km, dy_km)) + 360) % 360
        df['elevation'] = np.degrees(np.arctan2(df['rel_alt_km'], df['ground_dist_km']))
    
    return df

# --- サイドバー ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    st.markdown("---")
    use_filter = st.checkbox("Noise Filter", value=True)
    color_mode = st.radio("Path Color", ["Altitude", "RSSI"])
    pitch = st.slider("3D Map Pitch (Tilt)", 0, 60, 45)

# --- 読み込み ---
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        df = process_data(raw, use_filter)
    except Exception as e: st.error(e); st.stop()
else:
    try:
        raw = pd.read_csv('flight_data.csv')
        df = process_data(raw, use_filter)
    except: st.info("Upload CSV file."); st.stop()

if len(df) == 0: st.error("No Data."); st.stop()

# スライダー
slider_idx = st.slider("Time Scrubbing", 0, len(df) - 1, len(df) - 1, 1)
cur = df.iloc[slider_idx]
hist = df.iloc[:slider_idx+1]
gmap_url = f"https://www.google.com/maps/search/?api=1&query={cur['lat']},{cur['lon']}"

# ステータス判定
max_alt_idx = df['altitude'].argmax()
burst_alt = df.iloc[max_alt_idx]['altitude']
start_alt = df['altitude'].iloc[0]

if slider_idx <= max_alt_idx:
    if (cur['altitude'] - start_alt) < 10:
        status, s_col = "STANDBY", "#607D8B"
    else:
        status, s_col = "ASCENT", "#D32F2F"
    burst_disp = "---- m"
else:
    if (cur['altitude'] - start_alt) < 100:
        status, s_col = "SPLASHDOWN", "#0288D1"
    else:
        status, s_col = "DESCENT", "#303F9F"
    burst_disp = f"{burst_alt:.0f} m"

# --- 上部レイアウト ---
c1, c2 = st.columns([1, 3])
with c1:
    kml_data = generate_kml(df)
    st.download_button("Get KML", kml_data, "flight.kml")
    cols = ['timestamp', 'lat', 'lon', 'altitude', 'rssi', 'climb_rate', 'ground_speed']
    csv = df[[c for c in cols if c in df.columns]].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "data.csv")

with c2:
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Burst Alt", burst_disp)
    cc2.metric("Max Speed", f"{hist['climb_rate'].max():.1f} m/s")
    cc3.metric("Max Dist", f"{hist['ground_dist_km'].max():.2f} km")
    cc4.markdown(f"<div style='background:{s_col};color:white;padding:10px;text-align:center;border-radius:4px;font-weight:bold;'>{status}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- メインレイアウト ---
col_map, col_track, col_chart = st.columns([2.7, 1.8, 2.5])

with col_map:
    tab2d, tab3d = st.tabs(["2D Map", "3D View"])
    
    with tab2d:
        fig = go.Figure()
        if "RSSI" in color_mode and 'rssi' in hist.columns:
            fig.add_trace(go.Scattermapbox(
                lat=hist['lat'], lon=hist['lon'], mode='markers+lines',
                marker=dict(size=4, color=hist['rssi'], colorscale='RdYlBu', cmin=-120, cmax=-40),
                name='Path'
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=hist['lat'], lon=hist['lon'], mode='lines',
                line=dict(width=4, color='cyan'), name='Path'
            ))
        fig.add_trace(go.Scattermapbox(
            lat=[cur['lat']], lon=[cur['lon']], mode='markers',
            marker=dict(size=12, color='orange'), name='Pos'
        ))
        if slider_idx > max_alt_idx:
            bp = df.iloc[max_alt_idx]
            fig.add_trace(go.Scattermapbox(
                lat=[bp['lat']], lon=[bp['lon']], mode='markers',
                marker=dict(size=10, color='red'), name='Burst'
            ))
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=cur['lat'], lon=cur['lon']), zoom=10),
            height=450, margin=dict(l=0,r=0,t=0,b=0), showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"[Open Google Maps]({gmap_url})")

    with tab3d:
        path_data = hist[['lon', 'lat', 'altitude']].values.tolist()
        layer_path = pdk.Layer(
            "PathLayer", data=[{"path": path_data}],
            get_path="path", get_color=[255, 215, 0], width_min_pixels=3
        )
        layer_pos = pdk.Layer(
            "ScatterplotLayer", data=pd.DataFrame([cur]),
            get_position=["lon", "lat", "altitude"],
            get_color=[255, 165, 0], get_radius=200, pickable=True
        )
        view_state = pdk.ViewState(
            latitude=cur['lat'], longitude=cur['lon'],
            zoom=10, pitch=pitch, bearing=0
        )
        r = pdk.Deck(
            layers=[layer_path, layer_pos],
            initial_view_state=view_state,
            map_style="light", tooltip={"text": "Alt: {altitude}m"}
        )
        st.pydeck_chart(r)
        st.caption("※ [Shift]+Drag to rotate")

with col_track:
    st.subheader("Tracking")
    c_az, c_el = st.columns(2)
    c_az.metric("Azimuth", f"{cur['azimuth']:.1f}°", "North=0°")
    c_el.metric("Elevation", f"{cur['elevation']:.1f}°", "Up=+")
    st.divider()
    st.caption("Distance Info")
    st.metric("Horizontal Dist", f"{cur['ground_dist_km']:.2f} km")
    st.metric("Direct Dist (Slant)", f"{cur['slant_dist_km']:.2f} km")
    st.divider()
    st.metric("Altitude", f"{cur['altitude']:.0f} m")
    v = 0.0 if np.isnan(cur['climb_rate']) else cur['climb_rate']
    h = 0.0 if np.isnan(cur['ground_speed']) else cur['ground_speed']
    st.metric("Vert Spd", f"{v:.1f} m/s")
    st.metric("Gnd Spd", f"{h:.1f} m/s")

with col_chart:
    st.subheader("Logs")
    
    # 1. Altitude
    fig_alt = px.line(hist, x='timestamp', y='altitude', title="Altitude (m)")
    fig_alt.update_traces(line_color='#4CAF50')
    if slider_idx > max_alt_idx:
        fig_alt.add_hline(y=burst_alt, line_dash="dash", line_color="red")
    fig_alt.update_layout(height=200, margin=dict(t=30,b=0,l=0,r=0))
    st.plotly_chart(fig_alt, use_container_width=True)

    # 2. Speed
    fig_spd = go.Figure()
    fig_spd.add_trace(go.Scatter(x=hist['timestamp'], y=hist['climb_rate'], name='Vert', line=dict(color='#FF5722')))
    fig_spd.add_trace(go.Scatter(x=hist['timestamp'], y=hist['ground_speed'], name='Gnd', line=dict(color='#9C27B0')))
    fig_spd.update_layout(title="Speed (m/s)", height=200, margin=dict(t=30,b=0,l=0,r=0), showlegend=True)
    st.plotly_chart(fig_spd, use_container_width=True)

    # 3. Signal Strength (追加機能)
    if 'rssi' in hist.columns:
        fig_rssi = px.line(hist, x='timestamp', y='rssi', title="Signal Strength (dBm)")
        fig_rssi.update_traces(line_color='#2196F3')
        fig_rssi.update_layout(height=200, margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_rssi, use_container_width=True)

st.divider()
st.dataframe(df.round(2), use_container_width=True)

import os

filepath = 'd:\\IIT\\Final Year Project\\FishSense\\dashboard\\app.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start of the sections we want to modify.
# We know the block starts at 1384 (0-indexed 1383) with "#============================================================================"
# followed by "# LOCATION INPUT (MAIN AREA)".
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "# LOCATION INPUT (MAIN AREA)" in line:
        start_idx = i - 1
        break

for i in range(start_idx, len(lines)):
    if "# Calculate derived features" in lines[i]:
        end_idx = i - 1
        break

if start_idx == -1 or end_idx == -1:
    print("Could not find boundaries!")
    print(f"Start: {start_idx}, End: {end_idx}")
    exit(1)

new_layout = """#============================================================================
# MODE SELECTION
#============================================================================
st.markdown("### 🎛️ How would you like to predict?")
col1, col2 = st.columns(2)
with col1:
    quick_mode = st.button("🎯 Quick Prediction", type="primary", use_container_width=True)
with col2:
    live_mode = st.button("🛰️ Live Ocean Data", use_container_width=True)

# Store mode in session state
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Live Ocean Data"

if quick_mode:
    st.session_state.input_mode = "Quick Prediction"
elif live_mode:
    st.session_state.input_mode = "Live Ocean Data"

input_mode = st.session_state.input_mode

# Mode description
if input_mode == "Quick Prediction":
    st.info("💡 **Quick Prediction**: Perfect when you know the ocean conditions or want to test different scenarios.")
else:
    st.info("💡 **Live Ocean Data**: Automatically fetches real satellite data for your location. May use historical data if current data unavailable.")

st.markdown("---")

# Initialize default values (will be overridden by mode-specific inputs)
sst = 28.5
current_speed = 2.0
current_u = 1.0
current_v = 1.0

if input_mode == "Quick Prediction":
    # Manual input mode
    st.markdown("<h3 style='color: #ffffff;'>🌊 Ocean Conditions</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        sst = st.slider(
            "🌡️ Water Temperature (°C)",
            min_value=0.0,
            max_value=40.0,
            value=28.5,
            step=0.1,
            help="Optimal fishing temp is usually 20-30°C"
        )
    
    with col2:
        current_speed = st.slider(
            "🌊 Ocean Current Strength (m/s)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Moderate currents (2-4 m/s) are usually best for fishing"
        )
    
    # Advanced parameters
    with st.expander("🔬 Advanced Details (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            current_u = st.slider(
                "Current East-West (m/s)",
                min_value=-5.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Positive = Eastward flow"
            )
        
        with col2:
            current_v = st.slider(
                "Current North-South (m/s)",
                min_value=-5.0,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Positive = Northward flow"
            )
    st.markdown("---")


#============================================================================
# LOCATION INPUT (MAIN AREA)
#============================================================================
st.markdown("<h3 style='color: #ffffff;'>📍 Choose Your Fishing Location</h3>", unsafe_allow_html=True)

# --- POPULAR FISHING SPOTS PRESETS ---
fishing_spots = {
    "📍 Select a location...": None,
    "🐟 Negombo (West Coast)": {"lon": 79.0, "lat": 7.2},
    "🐟 Chilaw (West Coast)": {"lon": 79.1, "lat": 7.6},
    "🐟 Kalpitiya (Northwest)": {"lon": 79.2, "lat": 8.3},
    "🐟 Galle (South Coast)": {"lon": 79.9, "lat": 5.7},
    "🐟 Matara (South Coast)": {"lon": 80.4, "lat": 5.5},
    "🐟 Trincomalee (East Coast)": {"lon": 82.0, "lat": 8.6},
    "🐟 Batticaloa (East Coast)": {"lon": 82.2, "lat": 7.7},
    "🐟 Jaffna (North)": {"lon": 79.7, "lat": 10.0},
    "🐟 Mannar (Northwest)": {"lon": 79.2, "lat": 9.2},
    "🐟 Hambantota (South)": {"lon": 81.3, "lat": 5.7},
}

# Initialize session state for location
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = 6.5
if 'selected_lon' not in st.session_state:
    st.session_state.selected_lon = 79.2

selected_spot = st.selectbox(
    "🏝️ Quick Select: Popular Fishing Spots",
    options=list(fishing_spots.keys()),
    help="Choose a well-known fishing location to auto-fill coordinates"
)

# Update session state if a preset was selected
if fishing_spots[selected_spot] is not None:
    st.session_state.selected_lat = fishing_spots[selected_spot]["lat"]
    st.session_state.selected_lon = fishing_spots[selected_spot]["lon"]

# --- CLICKABLE MAP ---
st.markdown("<p style='color: #a78bfa; font-size: 0.9rem; margin-bottom: 0.3rem;'>🗺️ Or click on the map to select your fishing spot:</p>", unsafe_allow_html=True)

# Create a small clickable map
click_map = folium.Map(
    location=[7.5, 80.0],
    zoom_start=7,
    tiles='OpenStreetMap',
    width='100%',
    height=250
)

# Add study area boundary
folium.Rectangle(
    bounds=[[4.5, 77.0], [11.0, 83.5]],
    color='#8b5cf6',
    fill=False,
    weight=2,
    dash_array='5',
    popup='Study Area: Sri Lankan Waters'
).add_to(click_map)

# Add Sri Lanka outline for reference
folium.Rectangle(
    bounds=[[5.9, 79.5], [9.9, 81.9]],
    color='#ef4444',
    fill=True,
    fillColor='#ef4444',
    fillOpacity=0.1,
    weight=1,
    popup='Land Area (avoid)'
).add_to(click_map)

# Show current selection on the click map
folium.CircleMarker(
    location=[st.session_state.selected_lat, st.session_state.selected_lon],
    radius=8,
    color='#22c55e',
    fill=True,
    fillColor='#22c55e',
    fillOpacity=0.8,
    popup=f"Selected: [{st.session_state.selected_lat:.2f}, {st.session_state.selected_lon:.2f}]"
).add_to(click_map)

# Render interactive map (captures clicks)
map_data = st_folium(click_map, height=250, width=None, key="location_picker")

# Update coordinates from map click
if map_data and map_data.get('last_clicked'):
    clicked_lat = map_data['last_clicked']['lat']
    clicked_lon = map_data['last_clicked']['lng']
    # Clamp to study area
    clicked_lat = max(4.5, min(11.0, clicked_lat))
    clicked_lon = max(77.0, min(83.5, clicked_lon))
    st.session_state.selected_lat = round(clicked_lat, 2)
    st.session_state.selected_lon = round(clicked_lon, 2)

st.markdown("---")

# --- MANUAL COORDINATE INPUTS ---
st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>✏️ Or enter coordinates manually:</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    lon = st.number_input(
        "Longitude",
        min_value=77.0,
        max_value=83.5,
        value=st.session_state.selected_lon,
        step=0.1,
        format="%.2f",
        help="Sri Lankan waters: 77.0°E to 83.5°E"
    )
with col2:
    lat = st.number_input(
        "Latitude",
        min_value=4.5,
        max_value=11.0,
        value=st.session_state.selected_lat,
        step=0.1,
        format="%.2f",
        help="Sri Lankan waters: 4.5°N to 11.0°N"
    )

# Sync manual input back to session state
st.session_state.selected_lat = lat
st.session_state.selected_lon = lon

# Validate if coordinates are on land (Sri Lankan land mass)
# Sri Lanka roughly: 79.5°E to 81.9°E longitude, 5.9°N to 9.9°N latitude
if 79.5 <= lon <= 81.9 and 5.9 <= lat <= 9.9:
    st.warning("⚠️ **Warning**: These coordinates appear to be on LAND (Sri Lanka mainland). This model predicts ocean fishing zones. Please select coordinates in the ocean:\\n- **West coast**: Longitude < 79.5 (e.g., 79.2)\\n- **East coast**: Longitude > 81.9 (e.g., 82.1)")
    st.info("💡 **Suggested ocean locations**:\\n- West: Lon=79.2, Lat=6.5\\n- East: Lon=82.1, Lat=7.5\\n- Northwest: Lon=79.3, Lat=8.5")

st.markdown("---")

# Initialize session state for fetched data
if 'fetched_data' not in st.session_state:
    st.session_state.fetched_data = None

#============================================================================
# ACTION BUTTONS AND PREDICTION LOGIC
#============================================================================

if input_mode == "Quick Prediction":
    # Just the predict button using the manulay inputted values
    predict_button = st.button("🎯 Find Fishing Zones", type="primary", key="manual_predict", use_container_width=True)

else:
    # Live data mode
    fetch_button = st.button("🛰️ Fetch Live Ocean Data", type="primary", key="fetch_data", use_container_width=True)
    
    if not gee_initialized:
        err_detail = f" Error: {gee_init_error}" if gee_init_error else ""
        st.warning(f"Google Earth Engine not initialized — using Open-Meteo for live ocean data.{err_detail}")
    
    if fetch_button:
        if gee_initialized:
            # Try GEE first
            with st.spinner("🛰️ Fetching satellite data..."):
                ocean_data, error = fetch_realtime_data(lat, lon)
                if error:
                    st.warning(f"GEE fetch failed: {error}. Trying Open-Meteo...")
                    ocean_data = None
                else:
                    st.session_state.fetched_data = ocean_data
                    st.success(f"✅ Data fetched! {ocean_data['note']}")
        else:
            ocean_data = None
        
        # Fallback to Open-Meteo if GEE fails or isn't available
        if not st.session_state.get('fetched_data'):
            with st.spinner("🌊 Fetching from Open-Meteo..."):
                try:
                    meteo_currents = fetch_open_meteo_currents(lon, lat)
                    meteo_sst_data = fetch_open_meteo_sst(lon, lat)
                    
                    if meteo_currents or meteo_sst_data:
                        fallback_sst = meteo_sst_data['sst'] if meteo_sst_data else 28.5
                        fallback_speed = meteo_currents['current_speed'] if meteo_currents else 1.5
                        fallback_u = meteo_currents['u'] if meteo_currents else 0.9
                        fallback_v = meteo_currents['v'] if meteo_currents else 0.6
                        
                        st.session_state.fetched_data = {
                            'sst': fallback_sst,
                            'current_speed': fallback_speed,
                            'current_u': fallback_u,
                            'current_v': fallback_v,
                            'data_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'success': True,
                            'note': 'SST from Open-Meteo, Currents from Open-Meteo (live)'
                        }
                        st.success("✅ Live ocean data fetched from Open-Meteo!")
                    else:
                        st.error("Could not fetch data from Open-Meteo. Please try again or use Quick Prediction mode.")
                except Exception as e:
                    st.error(f"Open-Meteo fetch failed: {str(e)}. Try Quick Prediction mode.")
    
    # Display fetched data
    if st.session_state.fetched_data:
        data = st.session_state.fetched_data
        # Display fetched data in main area
        st.markdown("### 📊 Fetched Data")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌡️ Sea Temperature", f"{data['sst']:.2f}°C" if data['sst'] else "N/A")
        with col2:
            st.metric("🌊 Current Speed", f"{data['current_speed']:.2f} m/s" if data['current_speed'] else "N/A")
        
        st.caption(f"📅 Data from: {data['data_date']}")
        
        # Show data source note if available
        if 'note' in data:
            st.info(f"ℹ️ {data['note']}")
        
        # Use fetched data for prediction
        sst = data['sst'] if data['sst'] else 28.5
        current_speed = data['current_speed'] if data['current_speed'] else 2.0
        current_u = data['current_u'] if data['current_u'] else 1.0
        current_v = data['current_v'] if data['current_v'] else 1.0
        
        st.markdown("---")
        predict_button = st.button("🎯 Find Fishing Zones", type="primary", key="realtime_predict", use_container_width=True)
    else:
        st.info("👆 Click 'Fetch Live Ocean Data' to get real-time data for the selected location")
        predict_button = False
\n"""

lines[start_idx:end_idx] = [new_layout]

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("SUCCESS")

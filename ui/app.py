
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PathSnap | Map Matching",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap")
st.subheader("GPS Map Matching Dashboard")
st.write("Explore GPS trajectories and predicted road routes.")

# Fetch available trips
@st.cache_data(ttl=10)
def get_trips():
    response = requests.get(f"{API_URL}/trips", timeout=10)
    response.raise_for_status()
    return response.json()["trips"]

try:
    trips = get_trips()
except requests.RequestException:
    st.error("Cannot connect to FastAPI. Please start the backend first.")
    st.stop()

if not trips:
    st.warning("No trips are currently available.")
    st.stop()

trip_options = {
    f'{trip["trip_id"]} ({trip["split"]})': trip["trip_id"]
    for trip in trips
}

selected_label = st.selectbox(
    "Select a trip",
    list(trip_options.keys())
)

selected_trip = trip_options[selected_label]

if st.button("Run Map Matching", type="primary"):
    try:
        with st.spinner("Processing GPS trajectory..."):
            response = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

        st.session_state["prediction"] = result

    except requests.RequestException as error:
        st.error(f"Prediction failed: {error}")

result = st.session_state.get("prediction")

if result and result["trip_id"] == selected_trip:

    st.divider()
    st.subheader(f"Results for Trip {result['trip_id']}")

    col1, col2, col3 = st.columns(3)

    accuracy = result.get("top1_acc")
    continuity = result.get("continuity")
    latency = result.get("latency_ms")

    col1.metric(
        "Top-1 Accuracy",
        f"{accuracy:.2f}%" if accuracy is not None else "N/A"
    )

    col2.metric(
        "Route Continuity",
        f"{continuity:.2f}%" if continuity is not None else "N/A"
    )

    col3.metric(
        "Latency",
        f"{latency:.2f} ms" if latency is not None else "N/A"
    )

    gps = result.get("gps_coords", [])
    predicted = result.get("pred_coords", [])

    if gps or predicted:

        all_coords = gps + predicted

        center_lat = sum(p[1] for p in all_coords) / len(all_coords)
        center_lon = sum(p[0] for p in all_coords) / len(all_coords)

        map_view = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="OpenStreetMap"
        )

        if gps:
            folium.PolyLine(
                [(p[1], p[0]) for p in gps],
                color="blue",
                weight=3,
                opacity=0.8,
                tooltip="Original GPS trajectory"
            ).add_to(map_view)

            folium.CircleMarker(
                location=[gps[0][1], gps[0][0]],
                radius=5,
                color="blue",
                fill=True,
                tooltip="GPS Start"
            ).add_to(map_view)

        if predicted:
            folium.PolyLine(
                [(p[1], p[0]) for p in predicted],
                color="red",
                weight=5,
                opacity=0.8,
                tooltip="Predicted road route"
            ).add_to(map_view)

        folium.LayerControl().add_to(map_view)

        st.subheader("Trajectory Visualization")
        st.caption(
            "Blue: GPS trajectory | Red: Predicted road route"
        )

        st_folium(
            map_view,
            width=None,
            height=550,
            use_container_width=True
        )

    else:
        st.warning("The API returned no coordinates.")

    with st.expander("View raw prediction response"):
        st.json(result)

else:
    st.info("Select a trip and click Run Map Matching to view results.")


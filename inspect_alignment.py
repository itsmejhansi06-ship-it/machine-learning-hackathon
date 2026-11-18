from pyproj import Transformer
import folium

from src.data_loader import load_trip
from src.preprocessing.coordinates import project_trip
from src.preprocessing.road_network import build_arc_geometries
from src.preprocessing.route_alignment import align_route_to_track


TRIP_ID = "00000009"

trip = load_trip(TRIP_ID, split="student_test")

track, nodes, projection = project_trip(
    trip["track"],
    trip["nodes"]
)

to_wgs84 = Transformer.from_crs(
    projection["epsg"],
    "EPSG:4326",
    always_xy=True
)

arc_geometries = build_arc_geometries(
    nodes,
    trip["arcs"]
)

alignment = align_route_to_track(
    track,
    arc_geometries,
    trip["route"]
)

center = [
    trip["track"]["lat"].iloc[:30].mean(),
    trip["track"]["lon"].iloc[:30].mean()
]

m = folium.Map(location=center, zoom_start=17)

route_arc_ids = trip["route"][:20]
arc_lookup = arc_geometries.set_index("arc_id")

for position, arc_id in enumerate(route_arc_ids):
    geometry = arc_lookup.loc[arc_id, "geometry"]

    coordinates = []

    for x, y in geometry.coords:
        lon, lat = to_wgs84.transform(x, y)
        coordinates.append((lat, lon))

    folium.PolyLine(
        coordinates,
        color="blue",
        weight=5,
        opacity=0.7,
        tooltip=f"Route position {position}, arc {arc_id}"
    ).add_to(m)

for idx, row in trip["track"].head(30).iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=4,
        color="red",
        fill=True,
        fill_opacity=0.8,
        tooltip=f"GPS observation {idx}"
    ).add_to(m)

first = trip["track"].iloc[0]

folium.Marker(
    location=[first["lat"], first["lon"]],
    tooltip="First GPS observation",
    icon=folium.Icon(color="green")
).add_to(m)

m.save("results/alignment_inspection_00000009.html")

print("Map saved to results/alignment_inspection_00000009.html")

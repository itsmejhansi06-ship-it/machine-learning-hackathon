
from src.data_loader import load_trip

trip = load_trip("00000009", split="student_test")

arcs = trip["arcs"].set_index("arc_id")
route = trip["route"]

print("Checking first 20 consecutive route arcs...\n")

disconnected = []

for i in range(len(route) - 1):
    current_arc = arcs.loc[route[i]]
    next_arc = arcs.loc[route[i + 1]]

    current_end = current_arc["to_node"]
    next_start = next_arc["from_node"]

    if current_end != next_start:
        disconnected.append({
            "position": i,
            "current_arc": route[i],
            "next_arc": route[i + 1],
            "current_to_node": current_end,
            "next_from_node": next_start
        })

    if i < 20:
        print(
            f"{route[i]} -> {route[i+1]} | "
            f"{current_end} -> {next_start} | "
            f"{'CONNECTED' if current_end == next_start else 'DISCONNECTED'}"
        )

print("\nFull route connectivity:")
print("Total transitions:", len(route) - 1)
print("Disconnected transitions:", len(disconnected))

if disconnected:
    print("\nFirst 10 disconnected transitions:")
    for item in disconnected[:10]:
        print(item)
else:
    print("All consecutive route arcs are connected.")

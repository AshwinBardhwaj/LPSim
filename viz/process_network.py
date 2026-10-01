import csv, json

nodes = {int(r.get("index", i)): (float(r["x"]), float(r["y"])) for i, r in enumerate(csv.DictReader(open("data/networks/berkeley/nodes.csv")))}
features = []

for r in csv.DictReader(open("data/networks/berkeley/edges.csv")):
    u, v = int(r["u"]), int(r["v"])
    if u in nodes and v in nodes:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [nodes[u], nodes[v]]
            },
            "properties": {"edge_id": int(r["uniqueid"])}
        })

geojson = {"type": "FeatureCollection", "features": features}
with open("viz/network.json", "w") as f:
    json.dump(geojson, f)
print(f"Generated viz/network.json with {len(features)} road segments.")

#!/usr/bin/env python3
"""
LPSim Web Visualizer: serves network + simulation state via HTTP.
Run:
  python viz/server.py --network data/networks/sf_bay_area --port 8080
Then open http://localhost:8080 in a browser.
"""

import argparse
import csv
import json
import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def parse_wkt_linestring(wkt_str):
    """Parse WKT LINESTRING (x1 y1, x2 y2, ...) into [[lon, lat], ...]."""
    if not wkt_str:
        return None
    match = re.fullmatch(r"LINESTRING\s*\((.*)\)", wkt_str.strip(), re.I)
    if not match:
        return None
    try:
        points = []
        for pair in match[1].split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1])])
        return points if len(points) >= 2 else None
    except Exception:
        return None


def load_network_geojson(network_path):
    """Convert network CSV to GeoJSON for deck.gl rendering with true curve geometry."""
    nodes_file = os.path.join(network_path, "nodes.csv")
    edges_file = os.path.join(network_path, "edges.csv")

    nodes = {}
    with open(nodes_file) as f:
        reader = csv.DictReader(f)
        has_index = "index" in reader.fieldnames
        for i, row in enumerate(reader):
            idx = int(row["index"]) if has_index else i
            lon = float(row.get("lon", row.get("x", 0.0)))
            lat = float(row.get("lat", row.get("y", 0.0)))
            nodes[idx] = {"lon": lon, "lat": lat}

    edges_features = []
    with open(edges_file) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        has_uv = "u" in cols and "v" in cols
        for row in reader:
            if not has_uv:
                continue
            u, v = int(row["u"]), int(row["v"])
            if u not in nodes or v not in nodes:
                continue

            src = nodes[u]
            dst = nodes[v]

            # Parse intermediate curved coordinates if available; fallback to chord
            coords = parse_wkt_linestring(row.get("geometry", ""))
            if not coords:
                coords = [
                    [src["lon"], src["lat"]],
                    [dst["lon"], dst["lat"]]
                ]

            edges_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "id": int(row.get("uniqueid", 0)),
                    "lanes": int(float(row.get("lanes", 1))),
                    "speed_mph": float(row.get("speed_mph", 30)),
                    "length": float(row.get("length", 0)),
                }
            })

    return {
        "type": "FeatureCollection",
        "features": edges_features
    }


class VizHandler(SimpleHTTPRequestHandler):
    """Serve static files from viz/ and API endpoints."""

    def __init__(self, *args, network_geojson=None, **kwargs):
        self.network_geojson = network_geojson
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        if self.path == "/api/network":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(self.network_geojson).encode())
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready"}).encode())
        elif self.path == "/api/routes":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            routes_path = Path("viz/routes.geojson")
            if routes_path.exists():
                with routes_path.open("rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(json.dumps({"type": "FeatureCollection", "features": []}).encode())
        else:
            super().do_GET()


def make_handler(network_geojson):
    def handler(*args, **kwargs):
        return VizHandler(*args, network_geojson=network_geojson, **kwargs)
    return handler


def main():
    parser = argparse.ArgumentParser(description="LPSim Web Visualizer")
    parser.add_argument("--network", default="data/networks/sf_bay_area")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading network from {args.network}...")
    geojson = load_network_geojson(args.network)
    print(f"Loaded {len(geojson['features'])} edges")

    handler = make_handler(geojson)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"Visualizer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
"""
build_maps_ors.py

Builds one folium HTML map per weekday, showing the proposed delivery routes
with REAL road-snapped paths (via OpenRouteService), plus a toggleable layer
showing which stores are shed under the fuel-reduction proposal.

Run this on your own machine (needs internet access - it won't work in a
sandboxed/offline environment). Needs: pandas, folium, openrouteservice.

    pip install pandas folium openrouteservice

Files needed in the same folder:
    FoodstuffsLocations.csv
    baseline_monday.csv, baseline_tuesday.csv, ... baseline_saturday.csv
    monday_solution.csv, tuesday_solution.csv, ... saturday_solution.csv   (fuel-reduction versions, for the shed-store layer)
    build_maps_common.py

Output: maps/final_<Day>.html  (one per day)
"""

import time
import openrouteservice
from build_maps_common import DAYS, load_locations, load_day_data, build_map

# --- put your OpenRouteService API key here ---
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjlmNDA2Nzg0YmNiNTQwZjY5ZTdmMDgwODZiNDA3YjcxIiwiaCI6Im11cm11cjY0In0="

client = openrouteservice.Client(key=ORS_API_KEY)

# simple cache so re-running doesn't re-fetch identical routes
_cache = {}


def ors_route_coords(ordered_names, coord_map, retries=3):
    """Calls ORS once per route (multi-waypoint directions request), returns
    the actual road-snapped polyline as a list of (lat, lon) points."""
    key = tuple(ordered_names)
    if key in _cache:
        return _cache[key]

    # ORS wants [lon, lat] order (GeoJSON convention)
    coords_lonlat = [(coord_map[name][1], coord_map[name][0]) for name in ordered_names]

    for attempt in range(retries):
        try:
            result = client.directions(coords_lonlat, profile='driving-car', format='geojson')
            geometry = result['features'][0]['geometry']['coordinates']
            # geometry is [lon, lat] pairs -> convert to (lat, lon) for folium
            latlon = [(pt[1], pt[0]) for pt in geometry]
            _cache[key] = latlon
            time.sleep(1.5)  # stay well under the free-tier rate limit
            return latlon
        except Exception as e:
            print(f"    retry {attempt+1}/{retries} for route {ordered_names[:2]}...: {e}")
            time.sleep(3)

    # fallback: straight line, if ORS genuinely fails after retries
    print(f"    WARNING: falling back to straight line for {ordered_names}")
    fallback = [coord_map[name] for name in ordered_names]
    _cache[key] = fallback
    return fallback


def main():
    coord_map, locs = load_locations()

    for day in DAYS:
        print(f"Building {day}...")
        baseline, shed_stores = load_day_data(day)
        m = build_map(day, coord_map, baseline, shed_stores, ors_route_coords,
                      title_suffix="")
        m.save(f'maps/final_{day}.html')
        print(f"  {day}: {len(baseline)} routes, {len(shed_stores)} shed stores -> maps/final_{day}.html")


if __name__ == "__main__":
    main()

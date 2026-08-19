"""
Shared logic for building day-by-day trucking route maps.

Shows TWO separate, independently-solved plans as toggleable layers:
  - Baseline Routes: every store served, no shedding allowed
  - Fuel Reduction Routes: up to 20% of stores may be shed; the SOLVER
    RE-OPTIMIZES ALL ROUTES around whichever stores remain - this is NOT
    the baseline routes with some stops deleted, it's a different plan.

Only one route layer is meant to be viewed at a time (toggle between them),
since overlaying both would misleadingly suggest they share route structure.

Two draw modes:
  - straight_line: no internet needed, connects stops with straight lines (preview)
  - ors: calls OpenRouteService for real road-snapped routes (needs build_maps_ors.py)
"""
import pandas as pd
import folium

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

ROUTE_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324',
    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
    '#ff4500', '#2e8b57', '#1e90ff', '#daa520', '#8b008b', '#00ced1',
    '#dc143c', '#556b2f', '#ff1493', '#4682b4', '#b8860b', '#6a5acd',
]


def load_locations():
    locs = pd.read_csv('FoodstuffsLocations.csv')
    coord_map = {row['Supermarket']: (row['Lat'], row['Long']) for _, row in locs.iterrows()}
    wh = locs[locs['Type'] == 'Warehouse'].iloc[0]
    coord_map['Warehouse'] = (wh['Lat'], wh['Long'])
    return coord_map, locs


def load_day_data(day):
    """Loads BOTH plans separately - they are independently solved and will
    generally have different route compositions, even for stores both plans
    happen to deliver to."""
    baseline = pd.read_csv(f'baseline_{day.lower()}.csv')
    baseline = baseline[baseline['mode'] == 'owned'].reset_index(drop=True)

    fuel_file = f'{day.lower()}_solution.csv'
    fuel = pd.read_csv(fuel_file)
    fuel_routes = fuel[fuel['mode'] == 'owned'].reset_index(drop=True)
    shed_stores = [rid.replace('SKIP_', '') for rid in fuel[fuel['mode'] == 'skipped']['route_id']]

    return baseline, fuel_routes, shed_stores


def _add_route_layer(m, layer_name, routes_df, coord_map, get_route_coords, show):
    layer = folium.FeatureGroup(name=layer_name, show=show)
    for i, row in routes_df.iterrows():
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        stops = row['stops'].split(';')
        ordered_names = ['Warehouse'] + stops + ['Warehouse']

        line_coords = get_route_coords(ordered_names, coord_map)
        popup_text = (f"<b>{row['route_id']}</b><br>"
                      f"Stops: {' &rarr; '.join(stops)}<br>"
                      f"Pallets: {int(row['total_pallets'])}<br>"
                      f"Duration: {row['total_duration_sec']/3600:.2f} hrs<br>"
                      f"Cost: ${row['cost']:.2f}")
        folium.PolyLine(line_coords, color=color, weight=3.5, opacity=0.8,
                        popup=folium.Popup(popup_text, max_width=300)).add_to(layer)

        for stop in stops:
            lat, lon = coord_map[stop]
            folium.CircleMarker(
                [lat, lon], radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.9,
                popup=f"<b>{stop}</b><br>Route: {row['route_id']}",
            ).add_to(layer)
    layer.add_to(m)
    return layer


def build_map(day, coord_map, baseline, fuel_routes, shed_stores, get_route_coords, title_suffix=""):
    """get_route_coords(stop_names_in_order) -> list of (lat, lon) polyline points."""
    wh_lat, wh_lon = coord_map['Warehouse']
    m = folium.Map(location=[wh_lat, wh_lon], zoom_start=11, tiles='cartodbpositron')

    folium.Marker(
        [wh_lat, wh_lon],
        popup='Warehouse (Mt Roskill)',
        icon=folium.Icon(color='black', icon='home', prefix='fa'),
    ).add_to(m)

    # Two SEPARATE, mutually-exclusive route plans - only one shown by default
    _add_route_layer(m, f'BASELINE Routes ({len(baseline)}, no shedding)',
                     baseline, coord_map, get_route_coords, show=True)
    _add_route_layer(m, f'FUEL REDUCTION Routes ({len(fuel_routes)}, {len(shed_stores)} stores shed)',
                     fuel_routes, coord_map, get_route_coords, show=False)

    if shed_stores:
        shed_layer = folium.FeatureGroup(
            name=f'Shed Store Markers ({len(shed_stores)}) - pair with Fuel Reduction layer only',
            show=False)
        for store in shed_stores:
            lat, lon = coord_map[store]
            folium.Marker(
                [lat, lon],
                popup=f"<b>SHED:</b> {store}<br>(skipped under fuel-reduction plan)",
                icon=folium.Icon(color='gray', icon='ban', prefix='fa'),
            ).add_to(shed_layer)
        shed_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    title_html = f'''
        <div style="position: fixed; top: 10px; left: 60px; z-index: 9999;
                    background: white; padding: 8px 16px; border-radius: 6px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial; max-width: 420px;">
            <b style="font-size:16px;">{day} Delivery Routes{title_suffix}</b><br>
            <span style="font-size:12px; color:#555;">
                Baseline: {len(baseline)} routes, {baseline['total_pallets'].sum():.0f} pallets<br>
                Fuel Reduction: {len(fuel_routes)} routes, {fuel_routes['total_pallets'].sum():.0f} pallets, {len(shed_stores)} stores shed<br>
                <i>These are two separate plans - toggle one route layer at a time, not both together.</i>
            </span>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    return m

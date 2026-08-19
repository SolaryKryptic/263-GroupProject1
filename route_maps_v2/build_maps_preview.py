from build_maps_common import DAYS, load_locations, load_day_data, build_map

def straight_line_coords(ordered_names, coord_map):
    return [coord_map[name] for name in ordered_names]

coord_map, locs = load_locations()

for day in DAYS:
    baseline, fuel_routes, shed_stores = load_day_data(day)
    m = build_map(day, coord_map, baseline, fuel_routes, shed_stores, straight_line_coords,
                  title_suffix=" (PREVIEW - straight lines)")
    m.save(f'maps/preview_{day}.html')
    print(f"{day}: {len(baseline)} baseline routes, {len(fuel_routes)} fuel-reduction routes, "
          f"{len(shed_stores)} shed -> maps/preview_{day}.html")

def calculate_sustainability(area, small_plants, shrubs, trees):

    # ---- Carbon Absorption ----
    co2_from_trees = trees * 21
    co2_from_shrubs = shrubs * 5
    co2_from_small = small_plants * 1

    total_co2 = co2_from_trees + co2_from_shrubs + co2_from_small
    total_co2_tons = total_co2 / 1000  # convert to tons

    # ---- Cooling Estimate ----
    cooling_effect = (trees / 100) * 1.0
    cooling_effect = min(cooling_effect, 3.0)

    # ---- Sustainability Score ----
    density_score = min((trees / (area + 1e-6)) * 1000, 40)
    carbon_score = min(total_co2 / 50, 30)
    cooling_score = min(cooling_effect * 10, 30)

    sustainability_index = density_score + carbon_score + cooling_score
    sustainability_index = min(sustainability_index, 100)

    return {
        "co2_kg": total_co2,
        "co2_tons": total_co2_tons,
        "cooling_effect": cooling_effect,
        "sustainability_index": sustainability_index
    }
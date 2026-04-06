def estimate_plants_by_category(area):
    """
    Distributes plantable area into trees, shrubs, and small plants
    using realistic urban landscaping ratios.
    """

    # Safety for extremely small areas
    if area < 20:
        return {
            "category": "Very Small Area",
            "trees": 0,
            "shrubs": max(1, int(area / 5)),
            "small_plants": max(2, int(area / 1))
        }

    # Backyard / residential scale
    if area < 300:
        trees = max(1, int(area * 0.1 / 9))        # ~10% area for trees
        shrubs = max(2, int(area * 0.3 / 2))       # ~30% for shrubs
        small_plants = max(4, int(area * 0.6 / 0.5))

        return {
            "category": "Backyard / Residential",
            "trees": trees,
            "shrubs": shrubs,
            "small_plants": small_plants
        }

    # Open land / urban green space
    trees = int(area * 0.4 / 9)                    # 40% trees
    shrubs = int(area * 0.35 / 2)                  # 35% shrubs
    small_plants = int(area * 0.25 / 0.5)          # 25% small plants

    return {
        "category": "Open Land",
        "trees": trees,
        "shrubs": shrubs,
        "small_plants": small_plants
    }

import numpy as np
import cv2


def analyze_urban_density(mask, original_image_rgb=None):
    total_pixels    = mask.shape[0] * mask.shape[1]
    built_up_pixels = np.sum(mask == 0)
    built_up_percentage = (built_up_pixels / total_pixels) * 100

    # ── Detect EXISTING green cover from actual image colors ──────
    if original_image_rgb is not None:
        # Convert to HSV for reliable green detection
        img_resized = cv2.resize(original_image_rgb,
                                 (mask.shape[1], mask.shape[0]))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

        # Green range in HSV (covers trees, grass, vegetation)
        lower_green = np.array([30,  30,  30])
        upper_green = np.array([90, 255, 255])
        green_mask  = cv2.inRange(hsv, lower_green, upper_green)

        green_pixels      = np.sum(green_mask > 0)
        green_percentage  = (green_pixels / total_pixels) * 100
    else:
        # Fallback: use mask
        green_pixels     = np.sum(mask == 255)
        green_percentage = (green_pixels / total_pixels) * 100

    if built_up_percentage > 60:
        density_class = "High Urban Density"
    elif built_up_percentage > 30:
        density_class = "Medium Urban Density"
    else:
        density_class = "Low Urban Density"

    if green_percentage < 20:
        plantation_potential = "High"   # needs more planting
    elif green_percentage < 50:
        plantation_potential = "Moderate"
    else:
        plantation_potential = "Low"    # already well vegetated

    return {
        "green_percentage":    round(green_percentage, 2),
        "built_up_percentage": round(built_up_percentage, 2),
        "density_class":       density_class,
        "plantation_potential": plantation_potential
    }


def assess_greenery_sufficiency(green_percentage, area_sqm):
    """
    WHO standard: 9 sq.m green space per person
    Urban planning standard: 20-30% green cover = sufficient
    """
    if green_percentage >= 40:
        status  = "SUFFICIENT"
        color   = "green"
        message = (
            f"This area already has {green_percentage:.1f}% green cover - "
            f"well above the 30% urban planning standard. "
            f"Existing greenery is healthy and sufficient. "
            f"Focus on maintenance rather than new plantation."
        )
        recommendation = "Maintain existing green cover"

    elif green_percentage >= 20:
        status  = "MODERATE"
        color   = "orange"
        message = (
            f"This area has {green_percentage:.1f}% green cover - "
            f"meets minimum standards but below the ideal 30%. "
            f"Targeted plantation in sparse zones is recommended."
        )
        recommendation = "Targeted plantation in sparse zones"

    elif green_percentage >= 10:
        status  = "INSUFFICIENT"
        color   = "red"
        message = (
            f"This area has only {green_percentage:.1f}% green cover - "
            f"below the 20% minimum urban standard. "
            f"Active plantation is strongly recommended."
        )
        recommendation = "Active plantation strongly recommended"

    else:
        status  = "CRITICAL"
        color   = "darkred"
        message = (
            f"Critical: Only {green_percentage:.1f}% green cover detected. "
            f"This area is severely lacking vegetation. "
            f"Urgent plantation intervention is needed."
        )
        recommendation = "Urgent plantation intervention needed"

    return {
        "status":         status,
        "color":          color,
        "message":        message,
        "recommendation": recommendation,
        "who_standard":   green_percentage >= 9,
        "urban_standard": green_percentage >= 20
    }
# Shipping date calculator
import datetime
from app.utils.logging import log_info, log_error

def estimate_shipping(location: str) -> str:
    log_info(f"EstimateShipping input: {location}")

    shipping_map = {
            "new york": 3,
            "ny": 3,
            "california": 5,
            "ca": 5,
            "texas": 4,
            "tx": 4
        }
    
    # Default to 7 days if location isn't in our list
    days = shipping_map.get(location.lower(), 7)
    
    delivery_date = datetime.date.today() + datetime.timedelta(days=days)
    return f"Shipping to {location} takes approximately {days} days. Estimated arrival: {delivery_date}"


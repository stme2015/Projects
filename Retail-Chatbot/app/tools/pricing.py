# app/tools/pricing.py
def calculate_discount(input_str: str) -> str:
    try:
        # Splits "120,10" into [120.0, 10.0]
        price, discount = map(float, input_str.split(","))
        final_price = price - (price * discount / 100)
        return f"The discounted price is ${final_price:.2f}"
    except Exception:
        return "Error: Please provide input in 'price,discount' format."

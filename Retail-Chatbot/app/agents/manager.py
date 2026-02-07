from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from app.config import MODEL_NAME, GEMINI_API_KEY
from app.tools.product_search import product_search
from app.tools.shipping import estimate_shipping
from app.tools.pricing import calculate_discount
from app.agents.prompts import SYSTEM_PROMPT
import time

def get_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )

@tool
def product_search_tool(query: str) -> str:
    """Search the catalog for item details. Returns the name, price, and description."""
    return product_search(query)

@tool
def shipping_estimator_tool(location: str) -> str:
    """Calculates a delivery date. Input: location."""
    return estimate_shipping(location)

@tool
def pricing_calculator_tool(price_and_discount: str) -> str:
    """Calculates final price. Input: 'price,discount' (e.g. '120,10')."""
    return calculate_discount(price_and_discount)

def build_agent():
    tools = [product_search_tool, shipping_estimator_tool, pricing_calculator_tool]

    llm = get_llm()
    retry_delay = 15
    try:
        agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"Quota exceeded, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

    print("Agent created successfully")

    return agent 

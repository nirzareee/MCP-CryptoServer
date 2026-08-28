"""
This script demonstrates how to create a simple MCP server that fetches
the current price of a cryptocurrency using the CoinGecko API.
It uses the FastMCP library to create the server and handle requests.
"""
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Create our MCP server with a descriptive name
mcp = FastMCP("crypto_price_tracker")
async def _get(endpoint: str, params: dict) -> dict | None:
    """Shared helper for CoinGecko GET requests. Returns None on failure."""
    headers = {"User-Agent": "mcp-crypto-server/0.1", "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{COINGECKO_BASE_URL}{endpoint}",
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

# Now let's define our first tool - getting the current price of a cryptocurrency
@mcp.tool()
async def get_crypto_price(crypto_id: str, currency: str = "usd") -> str:
    """
    Get the current price of a cryptocurrency in a specified currency.
    
    Parameters:
    - crypto_id: The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
    - currency: The currency to display the price in (default: 'usd')
    
    Returns:
    - Current price information as a formatted string
    """
    # Construct the API URL
    url = f"{COINGECKO_BASE_URL}/simple/price"
    
    # Set up the query parameters
    params = {
        "ids": crypto_id,
        "vs_currencies": currency
    }
    
    try:
        # Make the API call
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the response
            data = response.json()
            
            # Check if we got data for the requested crypto
            if crypto_id not in data:
                return f"Cryptocurrency '{crypto_id}' not found. Please check the ID and try again."
            
            # Format and return the price information
            price = data[crypto_id][currency]
            return f"The current price of {crypto_id} is {price} {currency.upper()}"
            
    except httpx.HTTPStatusError as e:
        return f"API Error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error fetching price data: {str(e)}"

# You can add more tools here, following the same pattern as above
@mcp.tool()
async def get_portfolio(holdings: dict[str, float], currency: str = "usd") -> str:
    """Value a portfolio of cryptocurrency holdings.

    Args:
        holdings: Map of CoinGecko coin id to quantity held,
            e.g. {"bitcoin": 0.5, "ethereum": 3}. Use search_coin
            first if you are unsure of an id.
        currency: Fiat currency to value the portfolio in, e.g. "usd".
    """
    if not holdings:
        return "No holdings provided."

    data = await _get(
        "/simple/price",
        {
            "ids": ",".join(holdings),
            "vs_currencies": currency,
            "include_24hr_change": "true",
        },
    )

    if data is None:
        return "Could not reach the price API."

    unit = currency.upper()
    positions = []
    unpriced = []
    total = 0.0

    for coin_id, quantity in holdings.items():
        entry = data.get(coin_id)
        price = entry.get(currency) if entry else None

        if price is None:
            unpriced.append(coin_id)
            continue

        value = price * quantity
        total += value
        positions.append(
            {
                "id": coin_id,
                "quantity": quantity,
                "price": price,
                "value": value,
                "change": entry.get(f"{currency}_24h_change"),
            }
        )

    if not positions:
        return f"None of the requested coins could be priced: {', '.join(unpriced)}"

    positions.sort(key=lambda p: p["value"], reverse=True)

    lines = [f"Portfolio value: {total:,.2f} {unit}", ""]
    for p in positions:
        weight = p["value"] / total * 100
        line = (
            f"{p['id']}: {p['quantity']:g} @ {p['price']:,.2f} "
            f"= {p['value']:,.2f} {unit} ({weight:.1f}%)"
        )
        if p["change"] is not None:
            line += f" {p['change']:+.2f}% 24h"
        lines.append(line)

    if unpriced:
        lines.append("")
        lines.append(
            f"Not priced, excluded from the total: {', '.join(unpriced)}"
        )

    return "\n".join(lines)
# Run the MCP server
# This will start the server and listen for incoming requests
if __name__ == "__main__":
    mcp.run()
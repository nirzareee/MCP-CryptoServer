# MCP Crypto Server

An [MCP](https://modelcontextprotocol.io) server that gives Claude access to live
cryptocurrency market data. Built with the MCP Python SDK and the public CoinGecko API.

## Tools

| Tool | What it does |
| --- | --- |
| `get_price` | Current price of a coin, with 24h change |
| `get_market_data` | Market cap, volume, 24h range, distance from all-time high |
| `search_coin` | Resolve a name or ticker to a CoinGecko coin id |
| `get_trending` | Coins currently trending on CoinGecko |

No API key required — these are CoinGecko's free public endpoints.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/<your-username>/mcp-crypto-server.git
cd mcp-crypto-server
uv sync
```

> **Note on SDK versions:** this project pins `mcp<2`. Version 2 of the MCP Python SDK
> renamed `FastMCP` to `MCPServer` and swapped `httpx` for `httpx2`, so the imports here
> will not work on 2.x without changes.

## Testing locally

```bash
uv run mcp dev mcp_server.py
```

This opens the MCP Inspector in your browser, where each tool can be called directly.

## Connecting to Claude Desktop

Add the following to `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
    "mcpServers": {
        "crypto-price-tracker": {
            "command": "/ABSOLUTE/PATH/TO/uv",
            "args": [
                "--directory",
                "/ABSOLUTE/PATH/TO/mcp-crypto-server",
                "run",
                "mcp_server.py"
            ]
        }
    }
}
```

Both paths must be absolute — Claude Desktop does not inherit your shell's `PATH`.
Run `which uv` to find the first one. Quit Claude Desktop completely and reopen it;
the server appears under the Connectors menu.

## Roadmap

- [ ] Portfolio tracker — value a set of holdings in one batched request
- [ ] TTL cache to stay inside CoinGecko's free-tier rate limit
- [ ] Expose a watchlist as an MCP resource

## License

MIT

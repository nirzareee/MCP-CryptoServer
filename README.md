# MCP Crypto Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that gives
LLMs access to live cryptocurrency market data and portfolio analytics —
built as a reference implementation for production-quality MCP tool design.

**Python 3.11+ · asyncio · httpx · pytest · mypy · ruff · GitHub Actions**

- **4 tools** exposed to any MCP client, including Claude Desktop
- **134 tests** at 97% coverage, running in under a second
- **CI on every push** — lint, format, type check, and tests across two Python
  versions
- **Statistical analysis** — return correlation, annualised volatility, and
  maximum drawdown, implemented from first principles
- **Resilient by design** — response caching, graceful degradation on API
  failure, and typed error handling at the network boundary

---

## Tools

| Tool | What it does |
| --- | --- |
| `get_crypto_price` | Current price of a coin, with 24h change |
| `get_portfolio` | Values a set of holdings in a single batched request |
| `compare_coins` | Correlation, volatility, and drawdown for two coins |
| `search_coin` | Resolves a name or ticker to a CoinGecko id |

Data comes from CoinGecko's free public API. No key required.

## Architecture

```
mcp_server.py    tool definitions — fetch, format, catch
coingecko.py     all network access, plus caching
analytics.py     portfolio valuation (pure)
timeseries.py    price-series statistics (pure)
cache.py         TTL cache (pure)
```

`analytics`, `timeseries`, and `cache` import nothing beyond the standard
library — no HTTP, no clock, no globals. That is what lets most of the test
suite run without a mocking library: pass a dict, assert on a number.

The split is not decoration. Every function likely to be *wrong* — a
correlation coefficient, a drawdown, a cache expiry — lives somewhere it can
be tested exhaustively in microseconds. What remains in `coingecko.py` is thin
enough that mocking it is cheap.

## Design decisions

### Correlation is computed on returns, not prices

In a rising market, every coin's price trends upward. Correlating raw *prices*
therefore returns something close to 1.0 for almost any pair — a result driven
entirely by shared trend rather than by any relationship between the coins.

Differencing to daily returns removes the trend and asks the question that was
actually intended: do these two move together day to day?

### Series are aligned before they are compared

Two coins rarely have identical histories. One listed later; one has a gap
where the exchange went down. Zipping the raw arrays pairs Monday's bitcoin
against Tuesday's ethereum and produces a plausible-looking coefficient that
means nothing — with no error to signal it.

`timeseries.align` takes the intersection of dates first. Everything is also
bucketed to daily closes beforehand, because CoinGecko silently varies its
resolution by range (five-minute data for a day, hourly to ninety days, daily
beyond), and comparing an hourly series to a daily one compares different
things.

### Undefined statistics return `None`, not `0.0`

Correlate any coin against a stablecoin and the denominator collapses: a coin
pegged to a dollar has no variance to correlate with. Returning `0.0` would
assert "no relationship," which is a claim the data does not support.
Returning `None` says "unanswerable from this input," which is true, and the
tool then says so in plain language.

### Volatility annualises with 365, not 252

The equity convention is 252 trading days because equity markets close. Crypto
does not. Using 252 here would understate annualised volatility by roughly
18%.

### In-memory TTL cache, not Redis

CoinGecko's free tier allows on the order of 10–30 calls per minute. A single
conversation burns through them quickly — "what's my portfolio worth", then
"what about just bitcoin", then "check again" — and those are largely identical
requests seconds apart.

The cache is a dict with timestamps and a 60-second expiry.

An MCP server over stdio is one process serving one client. There is no second
process to share state with and nothing to survive a restart, so an external
store would add a dependency, a connection to fail, and a service to run in
order to solve a problem this deployment does not have.
`functools.lru_cache` is the closer call, but it has no expiry — cached prices
would be served forever, and staleness is the whole point.

Three details make it defensible rather than merely present:

- **The clock is injected.** `TTLCache(clock=...)` means expiry can be tested
  by assignment rather than by `sleep(61)`. It defaults to `time.monotonic`,
  not `time.time`, because wall-clock time can jump backwards on an NTP
  correction and make entries appear newer than they are.
- **Expired entries are retained.** When a live fetch fails, the client falls
  back to stale data rather than erroring — but `Fetched.staleness_note()`
  forces the age into the output. A tool that quietly hands stale prices to a
  model, which then states them as current, is a bug wearing resilience as a
  costume.
- **Every tool takes `refresh`.** The model can express user intent about
  freshness ("check again") instead of the server deciding for everyone.

### Errors are translated at the boundary

`coingecko.py` converts `httpx` exceptions into a single `CoinGeckoError` with
a message a person can act on — the 429 case names the rate limit explicitly.
Tools catch it and return the text. An exception escaping an MCP tool becomes
an opaque error the model cannot explain to the user.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/nirzareee/MCP-CryptoServer.git
cd MCP-CryptoServer
uv sync
```

> **SDK version:** this project pins `mcp<2`. Version 2 renamed `FastMCP` to
> `MCPServer` and replaced `httpx` with `httpx2`, so these imports do not work
> on 2.x unmodified.

## Testing

```bash
uv run pytest              # 134 tests, 97% coverage
uv run pytest --cov        # with coverage
```

```bash
uv run ruff check .        # lint
uv run ruff format --check .
uv run mypy .              # types
```

CI runs all four on every push, across Python 3.11 and 3.12.

A few tests are worth pointing at, because they exist to catch specific
mistakes rather than to raise a number:

- `test_stores_falsy_values` — a cached empty dict must not read as a miss. An
  implementation checking truthiness instead of `is not None` would refetch
  forever.
- `test_measured_from_running_peak_not_global_max` — a 50% crash that happens
  *before* a later all-time high is still a 50% crash. Anchoring on the global
  maximum would report the wrong figure.
- `test_positions_without_change_are_excluded_from_both_sides` — missing 24h
  data must not be treated as a 0% move, which would drag the weighted average
  toward zero and understate a real move.
- `test_matches_stdlib_implementation` — the correlation formula is checked
  against `statistics.correlation` rather than against its own output. An
  earlier version of this test used a hand-computed value that was wrong, and
  the failure caught it.

## Connecting to Claude Desktop

Add to `claude_desktop_config.json`:

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
                "/ABSOLUTE/PATH/TO/MCP-CryptoServer",
                "run",
                "mcp_server.py"
            ]
        }
    }
}
```

Both paths must be absolute — Claude Desktop does not inherit your shell's
`PATH`, which is the most common reason this step fails. `which uv` gives the
first. Quit Claude Desktop entirely and reopen; the server appears under
Connectors.

To exercise the tools without Claude Desktop:

```bash
uv run mcp dev mcp_server.py
```

## Roadmap

- **An MCP resource** exposing a watchlist. Resources are read directly by the
  client rather than called as functions; this server currently uses one of
  MCP's three primitives.
- **Rate-limit backoff.** The cache reduces call volume but does not retry; a
  429 with a `Retry-After` header is currently surfaced as an error message.

## Credits

The initial server scaffold came from the MCP tutorial in
[NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents).
Everything above — the module split, caching, analytics, tests, and CI — was
built on top of it.

## License

MIT

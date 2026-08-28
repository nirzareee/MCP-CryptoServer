"""A persisted list of coins the user is tracking.

This is the only module that touches the filesystem. It exists because MCP
resources are *read* by the client rather than called by the model, so the
watchlist needs somewhere to live between sessions — the server process does
not survive a restart, and neither does the cache.

Storage is a JSON file. Not SQLite, not a database: the data is an ordered
list of short strings that one process reads and writes. A schema and a
connection would buy nothing here.
"""

import json
import os
import re
import tempfile
from pathlib import Path

# CoinGecko ids are lowercase alphanumeric with hyphens. Validating on the way
# in keeps junk out of the stored file and out of the URLs built from it, and
# means a corrupted or hand-edited file cannot inject arbitrary text into an
# API request.
COIN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

# The watchlist is priced in a single batched request and rendered into the
# model's context, so it has to stay small enough for both.
MAX_ENTRIES = 50

DEFAULT_PATH = Path.home() / ".mcp-crypto-server" / "watchlist.json"


class WatchlistError(Exception):
    """Raised when a watchlist operation cannot be completed."""


def is_valid_coin_id(coin_id: str) -> bool:
    """Whether a string looks like a CoinGecko coin id."""
    return bool(COIN_ID_PATTERN.match(coin_id))


class Watchlist:
    """An ordered, de-duplicated set of coin ids backed by a JSON file.

    Order is insertion order rather than alphabetical: the coin added most
    recently is usually the one being thought about, and a stable order makes
    the resource's output diff-able between reads.
    """

    def __init__(self, path: Path | None = None) -> None:
        """
        Args:
            path: Where to store the file. Defaults to
                `~/.mcp-crypto-server/watchlist.json`. Injectable so tests can
                use a temporary directory rather than the real home directory.
        """
        self._path = path if path is not None else DEFAULT_PATH

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[str]:
        """Read the stored coin ids.

        Returns an empty list when the file is missing, unreadable, or
        malformed. A corrupt watchlist should not take the server down: the
        cost of losing it is one re-entry, and the cost of crashing on startup
        is every other tool becoming unavailable.

        Entries that fail validation are dropped, so a hand-edited file cannot
        put arbitrary strings into an API request.
        """
        if not self._path.exists():
            return []

        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(raw, list):
            return []

        seen: set[str] = set()
        coins: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not is_valid_coin_id(item):
                continue
            if item in seen:
                continue
            seen.add(item)
            coins.append(item)

        return coins[:MAX_ENTRIES]

    def save(self, coins: list[str]) -> None:
        """Write the coin ids, replacing whatever was there.

        The write goes to a temporary file in the same directory and is then
        moved into place. `os.replace` is atomic on POSIX, so a crash or a
        full disk partway through leaves the previous file intact rather than
        a half-written one. Writing directly to the destination risks
        truncating a good watchlist into nothing.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                dir=self._path.parent,
                prefix=".watchlist-",
                suffix=".tmp",
                delete=False,
            )
            with handle as f:
                json.dump(coins, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(handle.name, self._path)
        except OSError as exc:
            raise WatchlistError(f"Could not save the watchlist: {exc}") from exc

    def add(self, coin_id: str) -> list[str]:
        """Add a coin, returning the updated list.

        Raises:
            WatchlistError: If the id is malformed or the list is full.
        """
        coin_id = coin_id.strip().lower()

        if not is_valid_coin_id(coin_id):
            raise WatchlistError(
                f"'{coin_id}' is not a valid CoinGecko id. Ids are lowercase "
                f"words joined by hyphens, such as 'bitcoin' or 'avalanche-2'."
            )

        coins = self.load()

        if coin_id in coins:
            return coins

        if len(coins) >= MAX_ENTRIES:
            raise WatchlistError(
                f"The watchlist holds at most {MAX_ENTRIES} coins. "
                f"Remove one before adding another."
            )

        coins.append(coin_id)
        self.save(coins)
        return coins

    def remove(self, coin_id: str) -> list[str]:
        """Remove a coin, returning the updated list.

        Raises:
            WatchlistError: If the coin is not on the list. Silently doing
                nothing would let the model report a removal that never
                happened.
        """
        coin_id = coin_id.strip().lower()
        coins = self.load()

        if coin_id not in coins:
            raise WatchlistError(f"'{coin_id}' is not on the watchlist.")

        coins.remove(coin_id)
        self.save(coins)
        return coins

    def clear(self) -> None:
        """Empty the watchlist."""
        self.save([])

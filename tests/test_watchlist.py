"""Tests for watchlist persistence.

Every test gets a `tmp_path`, so nothing here touches the real
`~/.mcp-crypto-server/watchlist.json`. That is the point of making the path a
constructor argument: a suite that writes to the user's home directory is one
that cannot be run safely twice.
"""

import json
from pathlib import Path

import pytest

from watchlist import MAX_ENTRIES, Watchlist, WatchlistError, is_valid_coin_id


@pytest.fixture
def watchlist(tmp_path: Path) -> Watchlist:
    return Watchlist(path=tmp_path / "watchlist.json")


class TestValidation:
    @pytest.mark.parametrize(
        "coin_id", ["bitcoin", "avalanche-2", "0x", "ethereum-classic", "a"]
    )
    def test_accepts_real_id_shapes(self, coin_id: str) -> None:
        assert is_valid_coin_id(coin_id)

    @pytest.mark.parametrize(
        "coin_id",
        [
            "",
            "Bitcoin",  # uppercase
            "bit coin",  # space
            "-bitcoin",  # leading hyphen
            "bit/coin",  # path separator
            "../../etc/passwd",
            "bitcoin?vs_currency=usd",  # query injection
            "x" * 100,  # over length
        ],
    )
    def test_rejects_malformed_ids(self, coin_id: str) -> None:
        """Ids end up in file contents and in URLs, so both are guarded here."""
        assert not is_valid_coin_id(coin_id)


class TestAddRemove:
    def test_add_then_load(self, watchlist: Watchlist) -> None:
        watchlist.add("bitcoin")
        assert watchlist.load() == ["bitcoin"]

    def test_preserves_insertion_order(self, watchlist: Watchlist) -> None:
        for coin in ("solana", "bitcoin", "ethereum"):
            watchlist.add(coin)
        assert watchlist.load() == ["solana", "bitcoin", "ethereum"]

    def test_adding_twice_is_a_no_op(self, watchlist: Watchlist) -> None:
        watchlist.add("bitcoin")
        watchlist.add("bitcoin")
        assert watchlist.load() == ["bitcoin"]

    def test_normalises_case_and_whitespace(self, watchlist: Watchlist) -> None:
        """'  Bitcoin ' and 'bitcoin' are the same coin, not two entries."""
        watchlist.add("  Bitcoin ")
        assert watchlist.load() == ["bitcoin"]

    def test_rejects_malformed_id_with_a_usable_message(
        self, watchlist: Watchlist
    ) -> None:
        with pytest.raises(WatchlistError) as exc_info:
            watchlist.add("not a coin!")

        assert "bitcoin" in str(exc_info.value)  # names a valid example

    def test_remove(self, watchlist: Watchlist) -> None:
        watchlist.add("bitcoin")
        watchlist.add("ethereum")

        assert watchlist.remove("bitcoin") == ["ethereum"]

    def test_removing_an_absent_coin_raises(self, watchlist: Watchlist) -> None:
        """Silence would let the model report a removal that never happened."""
        with pytest.raises(WatchlistError):
            watchlist.remove("bitcoin")

    def test_clear(self, watchlist: Watchlist) -> None:
        watchlist.add("bitcoin")
        watchlist.clear()
        assert watchlist.load() == []


class TestCapacity:
    def test_rejects_beyond_the_cap(self, watchlist: Watchlist) -> None:
        """The list is priced in one request and rendered into context."""
        for i in range(MAX_ENTRIES):
            watchlist.add(f"coin-{i}")

        with pytest.raises(WatchlistError) as exc_info:
            watchlist.add("one-too-many")

        assert str(MAX_ENTRIES) in str(exc_info.value)

    def test_readding_at_capacity_is_allowed(self, watchlist: Watchlist) -> None:
        """A duplicate does not grow the list, so it must not be refused."""
        for i in range(MAX_ENTRIES):
            watchlist.add(f"coin-{i}")

        assert len(watchlist.add("coin-0")) == MAX_ENTRIES


class TestCorruptFiles:
    def test_missing_file_reads_as_empty(self, tmp_path: Path) -> None:
        assert Watchlist(path=tmp_path / "nothing.json").load() == []

    def test_invalid_json_reads_as_empty(self, tmp_path: Path) -> None:
        """A corrupt watchlist must not take the whole server down.

        Losing the list costs one re-entry; crashing on import costs every
        other tool.
        """
        path = tmp_path / "watchlist.json"
        path.write_text("{not json at all")

        assert Watchlist(path=path).load() == []

    def test_wrong_top_level_type_reads_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "watchlist.json"
        path.write_text('{"coins": ["bitcoin"]}')

        assert Watchlist(path=path).load() == []

    def test_drops_invalid_entries_but_keeps_good_ones(self, tmp_path: Path) -> None:
        """A hand-edited file must not smuggle arbitrary strings into URLs."""
        path = tmp_path / "watchlist.json"
        path.write_text(json.dumps(["bitcoin", "../etc/passwd", 42, "ethereum"]))

        assert Watchlist(path=path).load() == ["bitcoin", "ethereum"]

    def test_deduplicates_on_read(self, tmp_path: Path) -> None:
        path = tmp_path / "watchlist.json"
        path.write_text(json.dumps(["bitcoin", "bitcoin", "ethereum"]))

        assert Watchlist(path=path).load() == ["bitcoin", "ethereum"]

    def test_truncates_an_oversized_file(self, tmp_path: Path) -> None:
        path = tmp_path / "watchlist.json"
        path.write_text(json.dumps([f"coin-{i}" for i in range(200)]))

        assert len(Watchlist(path=path).load()) == MAX_ENTRIES


class TestPersistence:
    def test_survives_a_new_instance(self, tmp_path: Path) -> None:
        """The whole point: the process restarts, the watchlist does not."""
        path = tmp_path / "watchlist.json"
        Watchlist(path=path).add("bitcoin")

        assert Watchlist(path=path).load() == ["bitcoin"]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "watchlist.json"
        Watchlist(path=path).add("bitcoin")

        assert path.exists()

    def test_writes_readable_json(self, tmp_path: Path) -> None:
        """Stored as plain JSON so a user can inspect or edit it by hand."""
        path = tmp_path / "watchlist.json"
        Watchlist(path=path).add("bitcoin")

        assert json.loads(path.read_text()) == ["bitcoin"]

    def test_leaves_no_temporary_files_behind(self, tmp_path: Path) -> None:
        """The atomic write uses a temp file; it must not accumulate."""
        w = Watchlist(path=tmp_path / "watchlist.json")
        for coin in ("bitcoin", "ethereum", "solana"):
            w.add(coin)

        assert [p.name for p in tmp_path.iterdir()] == ["watchlist.json"]

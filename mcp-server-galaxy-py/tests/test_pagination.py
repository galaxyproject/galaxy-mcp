"""Tests for the shared pagination helpers used by the bounded list tools."""

import random

import pytest

from galaxy_mcp.server import _paginate, _pagination_info, _validate_pagination


class TestValidatePagination:
    """limit/offset rejection rules shared by every bounded tool."""

    def test_accepts_sane_window(self):
        _validate_pagination(10, 0, max_limit=100)
        _validate_pagination(100, 999, max_limit=100)

    @pytest.mark.parametrize("limit", [0, -1, -100])
    def test_rejects_non_positive_limit(self, limit):
        with pytest.raises(ValueError, match="limit must be at least 1"):
            _validate_pagination(limit, 0, max_limit=100)

    def test_rejects_absurd_limit_and_names_the_cap(self):
        with pytest.raises(ValueError, match="limit must be at most 100"):
            _validate_pagination(1_000_000, 0, max_limit=100)

    @pytest.mark.parametrize("offset", [-1, -50])
    def test_rejects_negative_offset(self, offset):
        with pytest.raises(ValueError, match="offset must be 0 or greater"):
            _validate_pagination(10, offset, max_limit=100)

    def test_a_tool_without_offset_is_not_told_to_use_one(self):
        with pytest.raises(ValueError) as excinfo:
            _validate_pagination(26, 0, max_limit=25, pageable=False)

        assert "offset" not in str(excinfo.value)
        assert "request 25 or fewer" in str(excinfo.value)


class TestPaginate:
    """Client-side slicing plus the metadata that describes the slice."""

    def test_first_page_of_many(self):
        page, info = _paginate(list(range(50)), limit=10, offset=0, noun="widgets")

        assert page == list(range(10))
        assert info.total_items == 50
        assert info.returned_items == 10
        assert info.limit == 10
        assert info.offset == 0
        assert info.has_next is True
        assert info.has_previous is False
        assert info.next_offset == 10
        assert info.previous_offset is None
        assert "10 of 50 widgets" in info.helper_text
        assert "offset=10" in info.helper_text

    def test_middle_page(self):
        page, info = _paginate(list(range(50)), limit=10, offset=20, noun="widgets")

        assert page == list(range(20, 30))
        assert info.has_next is True
        assert info.has_previous is True
        assert info.next_offset == 30
        assert info.previous_offset == 10

    def test_last_page_is_short_and_says_so(self):
        page, info = _paginate(list(range(25)), limit=10, offset=20, noun="widgets")

        assert page == [20, 21, 22, 23, 24]
        assert info.returned_items == 5
        assert info.has_next is False
        assert info.next_offset is None
        assert "This is the last page." in info.helper_text

    def test_exactly_one_full_page_has_no_next(self):
        _, info = _paginate(list(range(10)), limit=10, offset=0, noun="widgets")

        assert info.has_next is False
        assert info.next_offset is None

    def test_empty_result(self):
        page, info = _paginate([], limit=10, offset=0, noun="widgets")

        assert page == []
        assert info.total_items == 0
        assert info.returned_items == 0
        assert info.has_next is False
        assert info.has_previous is False
        assert info.next_offset is None

    def test_offset_past_the_end_says_so(self):
        page, info = _paginate(list(range(5)), limit=10, offset=99, noun="widgets")

        assert page == []
        assert info.total_items == 5
        assert info.has_next is False
        assert info.has_previous is True
        assert "past the end" in info.helper_text

    def test_walking_every_page_yields_every_item_once(self):
        items = list(range(37))
        seen: list[int] = []
        offset = 0
        while True:
            page, info = _paginate(items, limit=7, offset=offset, noun="widgets")
            seen.extend(page)
            if not info.has_next:
                break
            assert info.next_offset is not None
            offset = info.next_offset

        assert seen == items


class TestPaginationInfo:
    """Server-side paging reports the server's total, not the page length."""

    def test_short_page_with_more_available(self):
        info = _pagination_info(
            total_items=100, returned_items=10, limit=20, offset=0, noun="histories"
        )

        assert info.has_next is True
        assert info.next_offset == 10

    def test_empty_page_with_more_available_still_advances(self):
        info = _pagination_info(
            total_items=100, returned_items=0, limit=20, offset=0, noun="histories"
        )

        assert info.has_next is True
        assert info.next_offset == 20


class TestPaginateProperties:
    """Behavioural checks, so a wrong formula cannot pass by being restated."""

    @pytest.mark.parametrize("total", [0, 1, 2, 7, 25, 99, 123])
    @pytest.mark.parametrize("limit", [1, 2, 7, 20, 100])
    def test_walking_from_zero_sees_everything_once_and_stops(self, total, limit):
        items = list(range(total))
        seen: list[int] = []
        offset = 0
        for _ in range(total + 2):
            page, info = _paginate(items, limit=limit, offset=offset, noun="x")
            seen.extend(page)
            if not info.has_next:
                break
            assert info.next_offset is not None
            assert info.next_offset > offset
            offset = info.next_offset
        else:
            pytest.fail(f"walk did not terminate for total={total} limit={limit}")

        assert seen == items

    def test_metadata_always_describes_the_payload(self):
        random.seed(20260918)
        for _ in range(3000):
            total = random.choice([0, 1, 3, 10, 25, 99, 123, 1000])
            limit = random.choice([1, 3, 20, 50, 500])
            offset = random.choice([0, 1, 7, max(0, total - 1), total, total + 1, 5000])
            items = list(range(total))

            page, info = _paginate(items, limit=limit, offset=offset, noun="x")

            assert page == items[offset : offset + limit]
            assert info.returned_items == len(page)
            assert info.total_items == total
            assert info.has_next == (offset + len(page) < total)
            assert info.has_previous == (offset > 0)
            if info.has_next:
                assert info.next_offset is not None
                assert info.next_offset > offset
            else:
                assert info.next_offset is None

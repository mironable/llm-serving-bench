import pytest

from bench.stream_stats import summarize_stream


def test_ttft_itl_and_total():
    # First token at 0.2s, then 3 more tokens 0.05s apart.
    s = summarize_stream(10.0, [10.2, 10.25, 10.30, 10.35], tokens=4)
    assert s.ttft_s == pytest.approx(0.2)
    assert s.itl_mean_s == pytest.approx(0.05)
    assert s.total_s == pytest.approx(0.35)
    assert s.tokens == 4


def test_single_token_has_no_itl():
    s = summarize_stream(0.0, [0.1], tokens=1)
    assert s.itl_mean_s is None


def test_usage_tokens_override_chunk_count():
    # Server batched several tokens per chunk: ITL spreads total decode time over all tokens.
    s = summarize_stream(0.0, [0.1, 0.3], tokens=5)
    assert s.itl_mean_s == pytest.approx(0.2 / 4)


def test_empty_stream_raises():
    with pytest.raises(ValueError):
        summarize_stream(0.0, [], tokens=0)

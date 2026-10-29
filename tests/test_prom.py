from bench.prom import parse_prometheus, server_gauges

SAMPLE = """\
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="model"} 3.0
vllm:kv_cache_usage_perc{model_name="model"} 0.42
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="model"} 5
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="model"} 9
vllm:time_to_first_token_seconds_sum{model_name="model"} 1.5
vllm:time_to_first_token_seconds_count{model_name="model"} 9
weird_metric{label="has space inside"} 7
"""


def test_parse_drops_labels_and_keeps_values():
    m = parse_prometheus(SAMPLE)
    assert m["vllm:num_requests_waiting"] == 3.0
    assert m["vllm:kv_cache_usage_perc"] == 0.42
    assert m["vllm:time_to_first_token_seconds_sum"] == 1.5
    assert m["weird_metric"] == 7


def test_same_name_samples_are_summed():
    m = parse_prometheus(SAMPLE)
    assert m["vllm:time_to_first_token_seconds_bucket"] == 14


def test_server_gauges_per_server():
    vllm = server_gauges(parse_prometheus(SAMPLE), "vllm")
    assert vllm == {"queue_depth": 3.0, "kv_usage": 0.42}
    sglang = server_gauges(
        parse_prometheus("sglang:num_queue_reqs 2\nsglang:token_usage 0.7\n"), "sglang"
    )
    assert sglang == {"queue_depth": 2.0, "kv_usage": 0.7}


def test_missing_metric_is_none():
    assert server_gauges({}, "vllm") == {"queue_depth": None, "kv_usage": None}

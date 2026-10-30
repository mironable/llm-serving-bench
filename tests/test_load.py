from bench.load import messages_for, request_body, shared_system_prompt


def test_shared_prompt_is_deterministic_and_long():
    a, b = shared_system_prompt(), shared_system_prompt()
    assert a == b
    assert 1000 <= len(a.split()) <= 1400  # ~1500 tokens at 1.3 tokens per word


def test_messages_for_workloads():
    short = messages_for("short", 7)
    assert [m["role"] for m in short] == ["user"]
    shared = messages_for("shared-prefix", 7)
    assert [m["role"] for m in shared] == ["system", "user"]
    assert shared[1] == short[0]


def test_request_body_reasoning_fields():
    plain = request_body([{"role": "user", "content": "hi"}], 16, None, 512)
    assert "chat_template_kwargs" not in plain and "reasoning_effort" not in plain
    budget = request_body([], 16, "budget", 256)
    assert budget["chat_template_kwargs"] == {"enable_thinking": True}
    assert budget["thinking_token_budget"] == 256
    off = request_body([], 16, "off", 512)
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in off
    effort = request_body([], 16, None, 512, "low")
    assert effort["reasoning_effort"] == "low" and "chat_template_kwargs" not in effort

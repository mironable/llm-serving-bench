from bench.load import messages_for, shared_system_prompt


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

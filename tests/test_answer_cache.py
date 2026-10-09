from app.answer_cache import AnswerCache, normalize_query


def test_normalize_query_ignores_case_spacing_and_end_punctuation():
    assert normalize_query("  What was Apple's\n revenue in 2024? ") == "what was apple's revenue in 2024"
    assert normalize_query("Revenue 2024") != normalize_query("Revenue 2025")


def test_least_recently_used_entry_is_evicted():
    cache = AnswerCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "b" is now the least recently used
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3

import re
from collections import OrderedDict

# In-memory cache of finished answers, so a repeated question (most often one of the example
# questions in the UI) costs no Groq tokens. The index is loaded once at startup and never changes
# while the process runs, so entries never go stale; a new index means a new image and a new
# process, which starts with an empty cache. The cache is per process: it is lost on restart or
# scale-to-zero and not shared between replicas.
MAX_ENTRIES = 500


def normalize_query(query: str) -> str:
    # "What was Apple's revenue in 2024?" and "what was apple's  revenue in 2024" share an entry
    text = re.sub(r"\s+", " ", query.casefold()).strip()
    return text.rstrip(" ?!.")


class AnswerCache:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.entries = OrderedDict()

    def get(self, query: str):
        key = normalize_query(query)
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)  # least recently used entries are evicted first
        return self.entries[key]

    def put(self, query: str, value) -> None:
        key = normalize_query(query)
        self.entries[key] = value
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

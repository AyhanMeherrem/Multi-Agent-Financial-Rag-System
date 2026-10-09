# Keyword (BM25) search over the indexed chunks, used next to the vector search (see retrieve_nodes).
# Dense embeddings capture meaning but blur exact terms and figures: a question about "operating
# income" also pulls in paragraphs that only talk around it. BM25 scores chunks by the question's
# own words, weighted by how rare each word is, so the table row with that exact label ranks high.
# The index is built in memory from the Qdrant payloads at startup (under a second for ~4,000
# chunks), so the vector store does not have to change.
import json
import re

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from rank_bm25 import BM25Okapi

# Numbers keep their separators ("391,035", "8.04") so a figure is one token
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its of on or s than that the "
    "their this to was were what when which who why with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class KeywordIndex:
    def __init__(self, points):
        self.payloads, self.filters, corpus = [], [], []
        for p in points:
            text = json.loads(p.payload.get("_node_content") or "{}").get("text", "")
            self.payloads.append(p.payload)
            self.filters.append((p.payload.get("company"), p.payload.get("year"), p.payload.get("section"), str(p.id)))
            corpus.append(tokenize(text))
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, company: str = None, year: str = None, section: str = None, top_k: int = 8,
               node_ids: list | None = None) -> list[NodeWithScore]:
        # Same filters as the vector search: None means "any"; node_ids limits the search to those chunks
        allowed = set(node_ids) if node_ids is not None else None
        scores = self.bm25.get_scores(tokenize(query))
        candidates = [i for i, (c, y, s, point_id) in enumerate(self.filters)
                      if (company is None or c == company) and (year is None or y == year)
                      and (section is None or s == section) and (allowed is None or point_id in allowed)
                      and scores[i] > 0]
        best = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
        return [NodeWithScore(node=metadata_dict_to_node(self.payloads[i]), score=float(scores[i])) for i in best]

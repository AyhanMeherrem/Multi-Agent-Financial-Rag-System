from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode


class FakeLLM:
    # Same .chat(messages) interface as the LlamaIndex Groq client; returns a fixed reply
    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


def make_node(text: str, company="AAPL", year="2024", section="Item 8", score=0.5, node_id=None, **metadata):
    node = TextNode(text=text, id_=node_id or f"{company}-{year}-{section}-{text[:20]}",
                    metadata={"company": company, "year": year, "section": section, **metadata})
    return NodeWithScore(node=node, score=score)

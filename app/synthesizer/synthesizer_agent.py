import os
import sys
sys.path.append(".")
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"


from dotenv import load_dotenv
import phoenix as px
from llama_index.core import set_global_handler
from llama_index.llms.groq import Groq
from app.router.router_agent import load_index_from_qdrant, route_query_with_llm

load_dotenv()

print("Openning Arize Phoenix Dashboard on http://localhost:6006 ")
phoenix_session = px.launch_app()
set_global_handler("arize_phoenix")


def get_synthesizer_llm() -> Groq:
    """
    Initializes Groq LPU LLM Engine (llama-3.3-70b-versatile) for high-reasoning financial synthesis.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing or unconfigured in .env file!"
        )
    return Groq(model="llama-3.3-70b-versatile", api_key=api_key)

def synthesize_financial_answer(query_str: str, index):
    nodes, filters = route_query_with_llm(query_str, index)
    context_text = "\n\n".join([node.node.text for node in nodes])
    synthesizer_llm = get_synthesizer_llm()
    prompt = f"""You are an expert financial analyst assistant specializing in SEC 10-K filings.
                Answer the user's question based strictly on the provided financial context below.
                
                Context from 10-K Filings:
                --------------------------
                {context_text}
                --------------------------
                
                User Question: "{query_str}"
                
                Financial Analysis & Answer:"""
    response = synthesizer_llm.complete(prompt)
    print("LLM response: ", response.text)
    return response.text

if __name__ == "__main__":
    index = load_index_from_qdrant()
    
    while True:
        try:
            user_query = input("\n[Enter Question]: ").strip()
            if not user_query or user_query.lower() in ["exit", "quit"]:
                print("Exiting RAG session.")
                break
            
            synthesize_financial_answer(user_query, index)
        except KeyboardInterrupt:
            print("\nExiting session.")
            break

    try:
        px.close()
    except Exception:
        pass
import weaviate
import asyncio
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from weaviate.classes.init import AdditionalConfig, Timeout

from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.llm import OllamaLLM

import urllib.request
import json

from mcp.server.fastmcp import FastMCP

# --- NEW: agent memory imports ---
from neo4j_agent_memory import MemoryClient, MemorySettings

load_dotenv()

ollamaEndpoint = "http://localhost:11434/api/generate"

neo4jURI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
neo4jUser     = os.getenv("NEO4J_USER",     "neo4j")
neo4jPass     = os.getenv("NEO4J_PASS",     "dd11223344@")

weaviateHost  = os.getenv("WEAVIATE_HOST",  "localhost")
weaviatePort  = int(os.getenv("WEAVIATE_PORT", "8080"))

ollamaModel   = os.getenv("OLLAMA_MODEL",   "llama3.2")
ollamaTemp    = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

neo4jDB = GraphDatabase.driver(neo4jURI, auth=(neo4jUser, neo4jPass))

textTwocypherLLM = OllamaLLM(model_name=ollamaModel, model_params={"options": {"temperature": ollamaTemp}})
textTwocypherSchema = get_schema(neo4jDB, is_enhanced=True)

textTwocypherExamples = [
    "USER INPUT: 'what is the average rating for Beauty products?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'beauty' RETURN avg(r.star_rating) AS average_score",
    "USER INPUT: 'top 5 most reviewed products' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 5",
    "USER INPUT: 'show me reviews for mobile electronics with 5 stars' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'electronics' AND r.star_rating = 5 RETURN r.review_body LIMIT 10",
    "USER INPUT: 'which customer has written the most reviews for appliances?' QUERY: MATCH (c:Customer)-[:WROTE]->(r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'appliances' RETURN c.customer_id, count(r) AS total ORDER BY total DESC LIMIT 1"
]

textTwocypherRetriver = Text2CypherRetriever(
    driver = neo4jDB, llm = textTwocypherLLM, neo4j_schema = textTwocypherSchema, examples = textTwocypherExamples
)

memorySettings = MemorySettings(
    neo4j={
        "uri":      neo4jURI,
        "username": neo4jUser,
        "password": neo4jPass,
    },
    embedding={"provider": "sentence_transformers", "model": "BAAI/bge-small-en-v1.5"},  
)

MEMORY_SESSION_ID = os.getenv("MEMORY_SESSION_ID", "neo4j-agent-default")


async def NeoSearch(userQ, session_id=MEMORY_SESSION_ID):
    async with MemoryClient(memorySettings) as memory:

        await memory.short_term.add_message(
            session_id = session_id,
            role       = "user",
            content    = userQ,
        )

        memContext = await memory.get_context(
            query      = userQ,
            session_id = session_id,
        )

        t2cRes = ""
        try:
            result = textTwocypherRetriver.search(query_text=userQ)
            if result and result.items:
                t2cRes = "\n".join(item.content for item in result.items).strip()
        except Exception:
            t2cRes = ""

        if t2cRes:
            Description = (
                "You are an Amazon reviews assistant.\n"
                "Provide a natural language answer to the user's question using the provided database records.\n"
                "Be concise, direct, and conversational.\n\n"
                f"Memory Context:\n{memContext}\n\n"
                f"Database Data:\n{t2cRes}\n\n"
                f"User Question: {userQ}"
            )
        else:
            Description = (
                "You are a friendly Amazon reviews assistant.\n"
                "Provide a polite, conversational response to the user's message.\n\n"
                f"Memory Context:\n{memContext}\n\n"
                f"User Message: {userQ}"
            )

        try:
            ollamaRequest = {"model": ollamaModel, "prompt": Description, "stream": False}
            jsonData      = json.dumps(ollamaRequest).encode("utf-8")
            detailHead    = {"Content-Type": "application/json"}
            rep           = urllib.request.Request(ollamaEndpoint, data=jsonData, headers=detailHead, method="POST")

            with urllib.request.urlopen(rep) as r:
                ollamaRes  = r.read().decode("utf-8")
                resultData = json.loads(ollamaRes)
                response   = resultData.get("response", "").strip()
        except Exception as e:
            response = f"Ollama Failed: {e}"

        # 6. Store the assistant response
        await memory.short_term.add_message(
            session_id = session_id,
            role       = "assistant",
            content    = response,
        )

        await memory.long_term.add_entity(
            name        = userQ[:80],
            entity_type = "QUERY",
            attributes  = {"session": session_id},
        )

        return response


mcp = FastMCP("Neo4jAgent")


@mcp.tool()
async def search(question: str) -> str:
    """Processes your question naturally, retrieving graph data if relevant."""
    return await NeoSearch(question)


if __name__ == "__main__":
    if "-terminal" in sys.argv:
        async def chatInterface():
            print("Agent running in terminal mode. Type 'stop' to exit.\n")
            while True:
                try:
                    userQuestion = input("You: ").strip()
                    if userQuestion.lower() == "stop":
                        break
                    if not userQuestion:
                        continue

                    response = await NeoSearch(userQuestion)
                    print(f"Assistant: \n{response}\n")
                except KeyboardInterrupt:
                    break
        try:
            asyncio.run(chatInterface())
        except KeyboardInterrupt:
            pass
    else:
        print("Neo4j MCP server running:")
        mcp.run()
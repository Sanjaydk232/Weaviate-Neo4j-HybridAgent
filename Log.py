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
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.extraction import GLiNEREntityExtractor

load_dotenv()

ollamaEndpoint = "http://localhost:11434/api/generate"

neo4jURI     = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
neo4jUser    = os.getenv("NEO4J_USER",     "neo4j")
neo4jPass    = os.getenv("NEO4J_PASS",     "dd11223344@")

weaviateHost = os.getenv("WEAVIATE_HOST",  "localhost")
weaviatePort = int(os.getenv("WEAVIATE_PORT", "8080"))

ollamaModel  = os.getenv("OLLAMA_MODEL",   "llama3.2")
ollamaTemp   = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

neo4jDB = GraphDatabase.driver(neo4jURI, auth=(neo4jUser, neo4jPass))

textTwocypherLLM    = OllamaLLM(model_name=ollamaModel, model_params={"options": {"temperature": ollamaTemp}})
textTwocypherSchema = get_schema(neo4jDB, is_enhanced=True)

textTwocypherExamples = [
    "USER INPUT: 'what is the average rating for Beauty products?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'beauty' RETURN avg(r.star_rating) AS average_score",
    "USER INPUT: 'top 5 most reviewed products' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 5",
    "USER INPUT: 'show me reviews for mobile electronics with 5 stars' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'electronics' AND r.star_rating = 5 RETURN r.review_body LIMIT 10",
    "USER INPUT: 'which customer has written the most reviews for appliances?' QUERY: MATCH (c:Customer)-[:WROTE]->(r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'appliances' RETURN c.customer_id, count(r) AS total ORDER BY total DESC LIMIT 1",
    "USER INPUT: 'Hey, my name is K, I want to know the top 5 most reviewed products' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 5",
    "USER INPUT: 'Hi! I am Sarah, can you show me the best rated kitchen products?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'kitchen' RETURN p.product_title, avg(r.star_rating) AS avg_rating, count(r) AS review_count ORDER BY avg_rating DESC LIMIT 10",
    "USER INPUT: 'My name is Alex and I only like 5 star products, show me top electronics' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'electronics' AND r.star_rating = 5 RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 10",
    "USER INPUT: 'Hey there, I prefer highly reviewed items, what are the top 5 most reviewed beauty products?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'beauty' RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 5",
    "USER INPUT: 'show me only 5 star reviewed products with more than 10 reviews' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE r.star_rating = 5 WITH p, count(r) AS review_count WHERE review_count > 10 RETURN p.product_title, review_count ORDER BY review_count DESC LIMIT 10",
    "USER INPUT: 'what products have more than 5 reviews and a 5 star rating?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE r.star_rating = 5 WITH p, count(r) AS review_count WHERE review_count > 5 RETURN p.product_title, review_count ORDER BY review_count DESC LIMIT 10",
    "USER INPUT: 'lowest rated products in home and garden' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'home' OR toLower(p.product_category) CONTAINS 'garden' RETURN p.product_title, avg(r.star_rating) AS avg_rating ORDER BY avg_rating ASC LIMIT 5",
    "USER INPUT: 'what categories are available?' QUERY: MATCH (p:Product) RETURN DISTINCT p.product_category ORDER BY p.product_category",
    "USER INPUT: 'how many products are in each category?' QUERY: MATCH (p:Product) RETURN p.product_category, count(p) AS product_count ORDER BY product_count DESC",
    "USER INPUT: 'which category has the most 5 star reviews?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE r.star_rating = 5 RETURN p.product_category, count(r) AS five_star_count ORDER BY five_star_count DESC LIMIT 5",
    "USER INPUT: 'how many reviews has customer 12345 written?' QUERY: MATCH (c:Customer {customer_id: '12345'})-[:WROTE]->(r:Review) RETURN count(r) AS total_reviews",
    "USER INPUT: 'show me the most active reviewers overall' QUERY: MATCH (c:Customer)-[:WROTE]->(r:Review) RETURN c.customer_id, count(r) AS total ORDER BY total DESC LIMIT 10",
    "USER INPUT: 'show me recent review text for top gift card products' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'sports' RETURN p.product_title, r.review_body, r.star_rating LIMIT 10",
    "USER INPUT: 'find reviews that mention battery life' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(r.review_body) CONTAINS 'battery' RETURN p.product_title, r.review_body, r.star_rating LIMIT 10",
]

textTwocypherRetriver = Text2CypherRetriever(
    driver=neo4jDB, llm=textTwocypherLLM, neo4j_schema=textTwocypherSchema, examples=textTwocypherExamples
)

def semanticWeaviate(userQ):
    try:
        with weaviate.connect_to_local(
            host=weaviateHost, port=weaviatePort,
            additional_config=AdditionalConfig(timeout=Timeout(init=30, query=300, insert=120))
        ) as client:
            reviews = client.collections.get("AmazonReview")
            weaviateResult = reviews.generate.near_text(query=userQ, limit=3, grouped_task=userQ)
            return weaviateResult.generative.text
    except Exception as e:
        return f"Weaviate Failed: {e}"

classifyModelUse = [
    "top", "latest", "count", "avg", "average", "sum",
    "rating", "category", "how many", "total", "customers", "products"
]

memorySettings = MemorySettings(
    neo4j={
        "uri":      neo4jURI,
        "username": neo4jUser,
        "password": neo4jPass,
    },
    embedding={"provider": "sentence_transformers", "model": "BAAI/bge-small-en-v1.5"},
)

MEMORY_SESSION_ID = None

extractor = GLiNEREntityExtractor.for_poleo()


async def HybridSearch(userQ, session_id=None):
    global MEMORY_SESSION_ID
    
    target_session = session_id if session_id is not None else MEMORY_SESSION_ID
    if not target_session:
        return "System Notification: No user is currently logged in. Please log in first."

    async with MemoryClient(memorySettings) as memory:

        try:
            extracted = await extractor.extract(userQ)
            for entity in extracted.entities:
                result = await memory.long_term.add_entity(
                    name=entity.name,
                    entity_type=entity.type,
                    attributes={
                        "confidence": entity.confidence,
                        "session_id": target_session,
                        "context": "Extracted user preference"
                    }
                )
                if isinstance(result, tuple):
                    entity_node, dedup = result
                else:
                    entity_node = result
        except Exception as e:
            print(f"[Extraction Warning]: {e}")

        await memory.short_term.add_message(
            session_id=target_session,
            role="user",
            content=userQ,
        )

        memContext = await memory.get_context(
            query=userQ,
            session_id=target_session,
            user_id=user_id,
        )

        long_term_context = ""
        try:
            lt_entities = await memory.long_term.search_entities(query=userQ, limit=3)
            if lt_entities:
                matched_facts = [
                    f"- {ent.name} ({ent.type})" 
                    for ent in lt_entities 
                    if getattr(ent, 'attributes', {}).get('session_id') == target_session
                ]
                if matched_facts:
                    long_term_context = "\n".join(matched_facts)
        except Exception as e:
            print(f"[Long Term Context Retrieval Error]: {e}")

        usingNeo4j = any(word in userQ.lower() for word in classifyModelUse)

        t2cR = ""
        if usingNeo4j:
            try:
                result = textTwocypherRetriver.search(query_text=userQ)
                if result and result.items:
                    t2cR = "\n".join(item.content for item in result.items).strip()
            except Exception:
                t2cR = ""
        else:
            t2cR = semanticWeaviate(userQ)

        if t2cR:
            dataB = "Graph Data (Neo4j)" if usingNeo4j else "Semantic Data (Weaviate)"
            Description = (
                "You are an Amazon reviews assistant.\n"
                "Provide a natural language answer to the user's question using the provided data.\n"
                "Be concise, direct, and conversational.\n\n"
                f"Long-Term User Preferences:\n{long_term_context if long_term_context else '- None registered.'}\n\n"
                f"Memory Context:\n{memContext}\n\n"
                f"{dataB}:\n{t2cR}\n\n"
                f"User Question: {userQ}"
            )
        else:
            Description = (
                "You are a natural language Amazon reviews assistant.\n"
                "Provide a polite, conversational response to the user's message.\n\n"
                f"Long-Term User Preferences:\n{long_term_context if long_term_context else '- None registered.'}\n\n"
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

        await memory.short_term.add_message(
            session_id=target_session,
            role="assistant",
            content=response,
        )

        await memory.long_term.add_entity(
            name        = userQ[:80],
            entity_type = "QUERY",
            attributes  = {"session": target_session, "backend": "neo4j" if usingNeo4j else "weaviate"},
        )

        return response


mcp = FastMCP("HybridMemoryAgent")

@mcp.tool()
async def login(username: str) -> str:
    global MEMORY_SESSION_ID
    user_id = username.strip().lower()
    if not user_id:
        return "Login Failed: Username string cannot be empty."
    MEMORY_SESSION_ID = user_id
    return f"Successfully logged in as '{MEMORY_SESSION_ID}'. Session activated."

@mcp.tool()
async def logoff() -> str:
    global MEMORY_SESSION_ID
    if not MEMORY_SESSION_ID:
        return "Logoff Ignored: No active user session detected."
    
    logged_out_user = MEMORY_SESSION_ID
    try:
        async with MemoryClient(memorySettings) as memory:
            await memory.short_term.clear_session(session_id=logged_out_user)
        MEMORY_SESSION_ID = None
        return f"User '{logged_out_user}' logged off successfully. Short-term history cleared."
    except Exception as e:
        return f"Logoff error during history truncation: {e}"


@mcp.tool()
async def search(question: str) -> str:
    return await HybridSearch(question)


if __name__ == "__main__":
    if "-terminal" in sys.argv:
        async def chatInterface():
            print("Terminal UI Started. Commands: 'login <name>', 'logoff', 'stop'")
            while True:
                try:
                    userQuestion = input("You: ").strip()
                    if userQuestion.lower() == "stop":
                        break
                    if not userQuestion:
                        continue
                    
                    if userQuestion.lower().startswith("login "):
                        uname = userQuestion[6:].strip()
                        login_response = await login(uname)
                        print(f"System: {login_response}\n")
                    elif userQuestion.lower() == "logoff":
                        logoff_response = await logoff()
                        print(f"System: {logoff_response}\n")
                    else:
                        response = await HybridSearch(userQuestion)
                        print(f"Assistant:\n{response}\n")
                except KeyboardInterrupt:
                    break
        try:
            asyncio.run(chatInterface())
        except KeyboardInterrupt:
            pass
    else:
        print("HybridMemoryAgent MCP server running:")
        mcp.run()
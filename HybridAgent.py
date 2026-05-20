import weaviate
import asyncio
import os
import re
import sys
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from weaviate.classes.init import AdditionalConfig, Timeout

from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.llm import OllamaLLM

from mcp.server.fastmcp import FastMCP

# ── ADDED: memory imports (from documentation) ────────────────────────────────
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.extraction import GLiNEREntityExtractor
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

neo4jURI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
neo4jUser     = os.getenv("NEO4J_USER",     "neo4j")
neo4jPass     = os.getenv("NEO4J_PASS",     "dd11223344@")

weaviateHost  = os.getenv("WEAVIATE_HOST",  "localhost")
weaviatePort  = int(os.getenv("WEAVIATE_PORT", "8080"))

ollamaModel   = os.getenv("OLLAMA_MODEL",   "llama3.2")
ollamaTemp    = float(os.getenv("OLLAMA_TEMPERATURE", "0"))


neo4jDB = GraphDatabase.driver(neo4jURI, auth=(neo4jUser, neo4jPass))

textTwocypherLLM = OllamaLLM(model_name=ollamaModel,model_params={"options": {"temperature": ollamaTemp}})
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

def semanticWeaviate(userQ):
    try:
        with weaviate.connect_to_local(host=weaviateHost, port=weaviatePort, additional_config=AdditionalConfig(timeout=Timeout(init=30, query=300, insert=120))) as client:
            reviews = client.collections.get("AmazonReview")
            weavateResult = reviews.generate.near_text(query = userQ, limit = 3, grouped_task = userQ)
            return weavateResult.generative.text
    except Exception as e:
        return f"Weavite Failed: {e}"

async def HybridAgent(userQ):
    classifyModelUse = ["top", "latest", "count", "avg", "average", "sum", "rating", "category", "how many", "total", "customers", "products"]
    usingNeo4j = False
    for classiedWords in classifyModelUse:
        if classiedWords in userQ.lower():
            usingNeo4j = True
            break
    if usingNeo4j == True:
        print(f"Running Neo4j")
        try:
            result = textTwocypherRetriver.search(query_text = userQ)
            return "\n".join(item.content for item in result.items)
        except Exception as e:
            return f"Text2Cypher Error: {e}"
    else:
        weaviateSemantic = semanticWeaviate(userQ)
        return str(weaviateSemantic)


mcp = FastMCP("HybridReviewAgent")

@mcp.tool()
async def hybridSearch(question: str) -> str:
    return await HybridAgent(question)

@mcp.tool()
def cypherSearch(question: str) -> str:
    try:
        result = textTwocypherRetriver.search(query_text=question)
        return "\n".join(item.content for item in result.items)
    except Exception as e:
        return f"Text2Cypher error: {e}"

@mcp.tool()
def semanticsearch(question: str) -> str:
    return semanticWeaviate(question)

if __name__ == "__main__":
    if "-terminal" in sys.argv:
        async def chatInterface():
            # ── ADDED: initialize memory (doc: initialize()) ──────────────────
            settings = MemorySettings(
                neo4j={"uri": neo4jURI, "username": neo4jUser, "password": neo4jPass}, embedding={"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"}
            )
            memory = MemoryClient(settings)
            await memory.connect()
            extractor = GLiNEREntityExtractor.for_poleo()

            # ── ADDED: greet + get/create user (doc: get_or_create_user()) ────
            name = input("What's your name? ").strip() or "Guest"

            users = await memory.long_term.search_entities(query=name, entity_types=["CUSTOMER"], limit=1)
            if users and users[0].name.lower() == name.lower():
                print(f"✓ Welcome back, {name}!")
                user_id = users[0].id
            else:
                user, _ = await memory.long_term.add_entity(name=name, entity_type="CUSTOMER")
                print(f"✓ Nice to meet you, {name}!")
                user_id = user.id

            session_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            print(f"\n💬 Chat started (Session: {session_id})")
            # ─────────────────────────────────────────────────────────────────

            while True:
                try:
                    userQuestion = input(f"{name}: ").strip()
                    if userQuestion.lower() == "return":
                        break
                    if not userQuestion:
                        continue

                    # ── ADDED: memory command (doc: show_memory_stats()) ──────
                    if userQuestion.lower() == "memory":
                        print("\n" + "="*50)
                        print("What I Remember About You")
                        print("="*50)
                        prefs = await memory.long_term.search_preferences(query="")
                        if prefs:
                            print("\n Preferences:")
                            for pref in prefs:
                                print(f"  • {pref.category}: {pref.preference}")
                        entities = await memory.long_term.search_entities(query="", limit=10)
                        if entities:
                            print("\n Products/Brands You've Mentioned:")
                            for e in entities:
                                if e.type in ["PRODUCT", "BRAND", "ORGANIZATION", "OBJECT"]:
                                    print(f"  • {e.name}")
                        print("="*50 + "\n")
                        continue
                    # ─────────────────────────────────────────────────────────

                    # ── ADDED: store message + extract entities + preferences ──
                    await memory.short_term.add_message(
                        session_id=session_id, role="user", content=userQuestion
                    )
                    extraction = await extractor.extract(userQuestion)
                    for entity in extraction.entities:
                        await memory.long_term.add_entity(name=entity.name, entity_type=entity.type)

                    brands = ["samsung", "apple", "lg", "sony", "amazon", "bosch", "whirlpool"]
                    for brand in brands:
                        if brand in userQuestion.lower() and any(w in userQuestion.lower() for w in ("like", "prefer", "love")):
                            pref = f"Likes {brand.title()} products"
                            await memory.long_term.add_preference(preference=pref, category="brand", confidence=0.85)
                            print(f"  📝 Learned: {pref}")
                    budget_match = re.search(r"\$(\d+)", userQuestion)
                    if budget_match:
                        pref = f"Budget around ${budget_match.group(1)}"
                        await memory.long_term.add_preference(preference=pref, category="budget", confidence=0.9)
                        print(f"  📝 Learned: {pref}")
                    # ─────────────────────────────────────────────────────────

                    # ── ADDED: resolve references using short-term memory ─────
                    referenceWords = ["first", "second", "third", "that", "you mentioned", "previous", "last"]
                    resolvedQuestion = userQuestion
                    if any(w in userQuestion.lower() for w in referenceWords):
                        conversation = await memory.short_term.get_conversation(session_id=session_id)
                        lastMessages = conversation.messages[-6:] if conversation.messages else []
                        productTitles = []
                        for m in lastMessages:
                            if m.role == "assistant":
                                for line in m.content.splitlines():
                                    if "product_title" in line:
                                        parts = line.split("'")
                                        if len(parts) >= 2:
                                            productTitles.append(parts[1])
                        if productTitles:
                            q = userQuestion.lower()
                            if "first" in q:    target = productTitles[0]
                            elif "second" in q: target = productTitles[1] if len(productTitles) > 1 else productTitles[0]
                            elif "third" in q:  target = productTitles[2] if len(productTitles) > 2 else productTitles[0]
                            elif "last" in q:   target = productTitles[-1]
                            else:               target = productTitles[0]
                            for ref in ["the first product", "the second product", "the third product", "the last product", "that product"]:
                                resolvedQuestion = resolvedQuestion.replace(ref, f'"{target}"')
                            if resolvedQuestion == userQuestion:
                                resolvedQuestion = f"{userQuestion} for {target}"
                    # ─────────────────────────────────────────────────────────

                    response = await HybridAgent(resolvedQuestion)
                    print(f"Response: \n{response}\n")

                    # ── ADDED: store assistant response ───────────────────────
                    await memory.short_term.add_message(
                        session_id=session_id, role="assistant", content=str(response)
                    )
                    # ─────────────────────────────────────────────────────────

                except KeyboardInterrupt:
                    break

            print("\n👋 Goodbye! Your preferences have been saved for next time.")
            await memory.close()

        try:
            asyncio.run(chatInterface())
        except KeyboardInterrupt:
            pass
    else:
        print("HybridReviewAgent MCP server:")
        mcp.run()
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

from mcp.server.fastmcp import FastMCP

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
            while True:
                try:
                    userQuestion = input("Question: ").strip()
                    if userQuestion.lower() == "return":
                        break
                    if not userQuestion:
                        continue
                    response = await HybridAgent(userQuestion)
                    print(f"Response: \n{response}\n")
                except KeyboardInterrupt:
                    break
        try:
            asyncio.run(chatInterface())
        except KeyboardInterrupt:
            pass
    else:
        print("HybridReviewAgent MCP server:")
        mcp.run()
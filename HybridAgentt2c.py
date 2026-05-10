import weaviate
import asyncio
import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
import urllib.request
import json
from weaviate.classes.init import AdditionalConfig, Timeout

from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.llm import OllamaLLM

load_dotenv()

neo4jURI = os.getenv("neo4jURI","bolt://localhost:7687")
neo4jAUTH = (os.getenv("neo4jUser", "neo4j")),(os.getenv("neo4jPass","dd11223344@"))

ollamaEndpoint = "http://localhost:11434/api/generate"  

neo4jDB = GraphDatabase.driver(neo4jURI, auth=neo4jAUTH)

textTwocypherLLM = OllamaLLM(model_name="llama3.2", model_params={"options": {"temperature": 0}})
textTwocypherSchema = get_schema(neo4jDB, is_enhanced=True)

textTwocypherExamples = [
    "USER INPUT: 'what is the average rating for Beauty products?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'beauty' RETURN avg(r.star_rating) AS average_score",
    "USER INPUT: 'top 5 most reviewed products' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 5",
    "USER INPUT: 'show me reviews for mobile electronics with 5 stars' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'electronics' AND r.star_rating = 5 RETURN r.review_body LIMIT 10",
    "USER INPUT: 'which customer has written the most reviews for appliances?' QUERY: MATCH (c:Customer)-[:WROTE]->(r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'appliances' RETURN c.customer_id, count(r) AS total ORDER BY total DESC LIMIT 1"]

textTwocypherRetriver = Text2CypherRetriever(
    driver = neo4jDB, llm = textTwocypherLLM, neo4j_schema = textTwocypherSchema, examples = textTwocypherExamples
)
    
def semanticWeaviate(userQ):
    try:
        with weaviate.connect_to_local(additional_config=AdditionalConfig(timeout=Timeout(init=30, query=300, insert=120))) as client:
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
            print("Response: \n")
            for item in result.items:
                print(f" {item.content}")
        except Exception as e:
            print(f"Text2Cypher Error: {e}")
    else:
        print(f"Running Weaviate")
        weaviateSemantic = semanticWeaviate(userQ)
        print(f"Response: \n{weaviateSemantic}\n")

if __name__ == "__main__":
    print ("Review agent: \n")
    async def chatInterface():
        while True:
            try:
                userQuestion = input("Question: ").strip()

                if userQuestion.lower() in ["return"]:
                    break
                
                if not userQuestion:
                    continue

                await HybridAgent(userQuestion)
                
            except KeyboardInterrupt:
                break
    try:
        asyncio.run(chatInterface())
    except KeyboardInterrupt:
        pass
    
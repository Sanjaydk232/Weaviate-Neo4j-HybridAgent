import asyncio
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.llm import OllamaLLM

from mcp.server.fastmcp import FastMCP

load_dotenv()

# Configuration
neo4jURI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
neo4jUser     = os.getenv("NEO4J_USER",     "neo4j")
neo4jPass     = os.getenv("NEO4J_PASS",     "dd11223344@")

ollamaModel   = os.getenv("OLLAMA_MODEL",   "llama3.2")
ollamaTemp    = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# Drivers & Setup
neo4jDB = GraphDatabase.driver(neo4jURI, auth=(neo4jUser, neo4jPass))

textTwocypherLLM = OllamaLLM(model_name=ollamaModel, model_params={"options": {"temperature": ollamaTemp}})
textTwocypherSchema = get_schema(neo4jDB, is_enhanced=True)

textTwocypherExamples = [
    "USER INPUT: 'what is the average rating for Beauty products?' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) WHERE toLower(p.product_category) CONTAINS 'beauty' RETURN avg(r.star_rating) AS average_score",
    "USER INPUT: 'top 5 most reviewed products' QUERY: MATCH (r:Review)-[:REVIEWS]->(p:Product) RETURN p.product_title, count(r) AS review_count ORDER BY review_count DESC LIMIT 5"
]

textTwocypherRetriver = Text2CypherRetriever(
    driver=neo4jDB, llm=textTwocypherLLM, neo4j_schema=textTwocypherSchema, examples=textTwocypherExamples
)

async def process_agent_turn(user_question: str) -> str:
    """Greedily searches the DB. If data is found, it explains it; otherwise, it handles it as conversation."""
    db_data = ""
    
    # 1. Always attempt database lookup
    try:
        db_results = textTwocypherRetriver.search(query_text=user_question)
        if db_results and db_results.items:
            db_data = "\n".join(item.content for item in db_results.items).strip()
    except Exception:
        # If text2cypher fails (common for casual inputs like "Hello"), 
        # we silently catch it and keep db_data empty
        db_data = ""

    # 2. If the database returned content, synthesize an explanation
    if db_data:
        synthesis_prompt = f"""You are an Amazon reviews assistant. Provide a natural language answer 
        to the user's question using the provided database records. Be concise, direct, and conversational.

        User Question: {user_question}
        Database Records:
        {db_data}

        Answer:"""
        return textTwocypherLLM.invoke(synthesis_prompt).content.strip()
        
    # 3. Fallback: If DB returned nothing or failed, treat it as a conversational chat response
    else:
        casual_prompt = f"""You are a friendly Amazon reviews assistant. Provide a polite, conversational response 
        to the user's message. If they are asking about data, let them know you couldn't find records matching that request.

        User: {user_question}
        Assistant:"""
        return textTwocypherLLM.invoke(casual_prompt).content.strip()

# FastMCP Setup
mcp = FastMCP("Neo4jEasyFlowAgent")  

@mcp.tool()
async def search(question: str) -> str:
    """Process a natural query or chat with the agent."""
    return await process_agent_turn(question)

if __name__ == "__main__":
    if "-terminal" in sys.argv:
        async def chatInterface():
            print("Agent loaded. Ready for easy-flow chat! Type 'Stop' to exit.\n")
            while True:
                try:
                    userQuestion = input("You: ").strip()
                    if userQuestion.lower() == "stop":
                        break
                    if not userQuestion:
                        continue
                    response = await process_agent_turn(userQuestion)
                    print(f"Assistant: {response}\n")
                except KeyboardInterrupt:
                    break
        try:
            asyncio.run(chatInterface())
        except KeyboardInterrupt:
            pass
    else:
        print("Neo4j MCP server running:")
        mcp.run()
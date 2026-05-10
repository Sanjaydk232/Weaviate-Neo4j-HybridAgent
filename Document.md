## Data ##

Kaggle: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset/data

For this agent specifically, I used only the three smallest .tsv files: amazon_reviews_us_Gift_Card_v1_00, amazon_reviews_us_Major_Appliances_v1_00, and amazon_reviews_us_Mobile_Electronics_v1_00, since I am only loading them locally.

## Neo4j ##

My import schema in Neo4j Desktop is set up as follows: (Customer)-[:WROTE]->(Review)-[:REVIEWS]->(Product)

All the .tsv files have the same name formatting, so each of the nodes and relations are imported with these properties:

(Customer):
customer_id

WROTE (Node Relations):
From = customer_id
To = review_id

(Review):
review_id
star_rating
helpful_votes
total_votes
verified_purchase
review_headline
review_body
review_date

REVIEWS:
From = review_id
To = product_id

(Product):
product_id
product_parent
product_title
product_category

I imported the three files mentioned above one by one locally.

## Weaviate ##

I followed the self-host instructions and hosted the database in Docker: https://docs.weaviate.io/weaviate/quickstart/local

I first created a collection called AmazonReview. Then I used nomic-embed-text and llama3.2 from Ollama to give the data meaning.

The following are the properties I vectorized:
customer_id
product_parent
product_title
star_rating
helpful_votes
total_votes
review_headline
review_body

Here, the "key value" was the review_body. Since I have set it up so Neo4j looks at the data’s relations to provide data-specific answers, I used semantic search to make Weaviate provide more personal answers like, "Why do customers dislike this certain product?"

## Code ##

Work flow:

classifyModelUse helps classify which tool to use to answer the user’s question. It contains words like average, top, etc., essentially words that focus on comparison and data search, like "which category has the highest average star ratings" or "find the top 3 customers who have written the most reviews overall."

HybridAgent() then uses classifyModelUse to identify if the user has used these words in their question, and if they did, it will use the Neo4j schema to answer the user’s question; else, it uses semantic search to answer. Weaviate here helps with questions like how people perceive certain products or summarizing the reviews of multiple products and comparing them to find the best product based on the reviews.

Neo4j Flow:
The agent first parses the question to generateCypher(), which contains detailed instructions that the agent should follow when attempting to generate the cypher, then returns a Cypher statement, which is printed in the console for my reference to ensure the right query is being parsed into cypherQuery(), where it returns the records returned by the Cypher and prints them in the console.

Weaviate Flow:
The agent parses the question into semanticWeaviate(), which connects to the local database, then generates the response using reviews.generate.near_text(), where reviews is the collection (data) I imported into Docker. Then it prints the result to the console.

Finally, using async, I created chatInterface(), which handles the chat interface, making it feel immersive instead of having to rerun the agent for every question.

## Comparisson ##

Ollama Cypher, Pros and Cons:

Pros:
- The big pro of using Ollama to write the Cypher is the freedom to answer almost all questions freely. My last agent required a certain structure from the user’s question, but using the LLM removes that restriction. For example, Ollama easily handles questions like, "Find customers who gave a 5-star rating to refrigerators but gave only one star to gift card products."

Cons:
- As I mentioned in the meeting, the LLM does not create accurate Cypher queries every time. As the types of questions grow, I would have to adapt the modelsGuide to reflect this and ensure accuracy, like often getting the relations wrong and creating new properties that are not in the original database. 
- It needs exact words; for example, if the user asks something regarding phones instead of Electronics as imported from the data, it will return an error.

VectorCypherRetriever, Pros and Cons:

Pros:
- Unlike the LLM approach, my last agent did not require the agent to use exact words. It will find relevant products or nodes using vector embeddings. 
- Using the retrievalQuery I had set up, I was also able to pull the neighboring nodes as well, which were then parsed to the LLM, giving more information to provide accurate results. 
- Since the retrievalQuery also has a predefined template, there is less chance of the generated Cypher being inaccurate.
Cons:
- It cannot handle questions that deal with the entire database, like "How many products have 2 stars?" as it only retrieves the top similar nodes. 
- Often, the returned Cypher would contain a large set of subgraphs that would be sent to the LLM, which increases costs.

Features | Ollama | Vector Retrieval

Best used for | Counts, Averages, Filtering | Recommendations (Similar to Weaviate)
Logic | LLM writes it | Pre-defined (retrievalQuery)
Accuracy | About the same for both (80%), just based on the question
Setup | Prompt Engineering (Easy to set up) | Requires vector embeddings and models (Requires a lot of time and proper structure)

## Link ##
https://docs.weaviate.io/weaviate/quickstart/local
https://neo4j.com/docs/python-manual/current/
https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
https://docs.python.org/3/library/urllib.request.html#module-urllib.request
https://docs.weaviate.io/weaviate/client-libraries/python/notes-best-practices

## Questions: ##

Which category has the highest average star rating: Major Appliance or Gift Card?
Do people have different expectations for quality when buying Gift Cards versus Major Appliances?
Show me the top 3 best-selling products in mobile electronics.
Show me reviews where people are talking about using a gift card for a birthday
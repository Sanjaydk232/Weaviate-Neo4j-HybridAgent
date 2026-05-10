import weaviate
from weaviate.classes.config import Configure, Property, DataType
import pandas as pd

# List of files to import
files = [
    "amazon_reviews_us_Gift_Card_v1_00.tsv",
    "amazon_reviews_us_Major_Appliances_v1_00.tsv",
    "amazon_reviews_us_Mobile_Electronics_v1_00.tsv"
]

# Step 1.1: Connect to your local Weaviate instance
with weaviate.connect_to_local() as client:
    
    # Optional: Delete if collection already exists to avoid duplicate imports while testing
    if client.collections.exists("AmazonReview"):
        client.collections.delete("AmazonReview")
        print("Existing collection deleted.")

    # Step 1.2: Create the collection with properties
    reviews = client.collections.create(
        name="AmazonReview",
        vector_config=Configure.Vectors.text2vec_ollama(
            api_endpoint="http://ollama:11434",
            model="nomic-embed-text",
        ),
        # This handles the RAG/GENERATION (summarizing the reviews)
        generative_config=Configure.Generative.ollama(
            api_endpoint="http://ollama:11434",
            model="llama3.2",
        ),
        properties=[
            Property(name="marketplace", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="customer_id", data_type=DataType.INT),
            Property(name="review_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="product_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="product_parent", data_type=DataType.INT),
            Property(name="product_title", data_type=DataType.TEXT),
            Property(name="product_category", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="star_rating", data_type=DataType.INT),
            Property(name="helpful_votes", data_type=DataType.INT),
            Property(name="total_votes", data_type=DataType.INT),
            Property(name="vine", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="verified_purchase", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="review_headline", data_type=DataType.TEXT),
            Property(name="review_body", data_type=DataType.TEXT),
            Property(name="review_date", data_type=DataType.TEXT, skip_vectorization=True),
        ]
    )

    # Step 1.3: Import the objects
    reviews = client.collections.use("AmazonReview")
    
    total_imported = 0
    # Use a fixed batch size to optimize speed
    with reviews.batch.fixed_size(batch_size=100) as batch:
        for file_path in files:
            print(f"Reading file: {file_path}")
            
            try:
                # CRITICAL: We are using nrows=1000 here to limit runtime during testing. 
                # Remove 'nrows=1000' once you are sure it works and are ready to import all rows!
                df = pd.read_csv(file_path, sep='\t', nrows=5000)
            except FileNotFoundError:
                print(f"Skipping {file_path} because the file was not found.")
                continue
                
            # Clean up potential missing numbers or texts to keep schema strict
            df["customer_id"] = pd.to_numeric(df["customer_id"], errors='coerce').fillna(0).astype(int)
            df["star_rating"] = pd.to_numeric(df["star_rating"], errors='coerce').fillna(0).astype(int)
            df["product_parent"] = pd.to_numeric(df["product_parent"], errors='coerce').fillna(0).astype(int)
            df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors='coerce').fillna(0).astype(int)
            df["total_votes"] = pd.to_numeric(df["total_votes"], errors='coerce').fillna(0).astype(int)
            df = df.fillna("")
            
            for _, row in df.iterrows():
                obj = {
                    "marketplace": str(row["marketplace"]),
                    "customer_id": int(row["customer_id"]),
                    "review_id": str(row["review_id"]),
                    "product_id": str(row["product_id"]),
                    "product_parent": int(row["product_parent"]),
                    "product_title": str(row["product_title"]),
                    "product_category": str(row["product_category"]),
                    "star_rating": int(row["star_rating"]),
                    "helpful_votes": int(row["helpful_votes"]),
                    "total_votes": int(row["total_votes"]),
                    "vine": str(row["vine"]),
                    "verified_purchase": str(row["verified_purchase"]),
                    "review_headline": str(row["review_headline"]),
                    "review_body": str(row["review_body"]),
                    "review_date": str(row["review_date"])
                }
                
                batch.add_object(properties=obj)
                total_imported += 1
                
    print(f"Successfully loaded {total_imported} objects into Weaviate.")
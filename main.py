import warnings
warnings.filterwarnings('ignore')

import os 
from datasets import load_dataset
import pandas as pd

from models import Listing, SearchResultItem

from pydantic import ValidationError

from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel

from openai import OpenAI

from IPython.display import display, HTML
import time



database_name = "airbnb_dataset"
collection_name = "listings_reviews"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MONGO_URI = os.environ.get("MONGO_URI")

text_embedding_field_name = "text_embeddings"
vector_search_index_name_text = "vector_index_text"


dataset = load_dataset("MongoDB/airbnb_embeddings", streaming=True, split="train")
dataset = dataset.take(100)
dataset_df = pd.DataFrame(dataset)


records = dataset_df.to_dict(orient='records')


# To handle catch `NaT` values
for record in records:
    for key, value in record.items():
        # Check if the value is list-like; if so, process each element.
        if isinstance(value, list):
            processed_list = [None if pd.isnull(v) else v for v in value]
            record[key] = processed_list
        else:
            if pd.isnull(value):
                record[key] = None


try:
    listings = [Listing(**record).dict() for record in records]
    print(listings[0].keys())
except ValidationError as e:
    print(e)    




def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = MongoClient(
            mongo_uri
        )
        print("Connection to MongoDB successful")
        return client
    
    except Exception as e:
        print(f"Connection to MongoDB failed: {str(e)}")
        return None



if not MONGO_URI:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(MONGO_URI)

if mongo_client:
    db = mongo_client.get_database(database_name)
    collection = db.get_collection(collection_name)

else:
    print("Failed to create MongoDB client.")


# collection.delete_many({})
# collection.insert_many(listings)
print("Data ingestion into MongoDB completed")



vector_search_index_model = SearchIndexModel(
    definition={
        "mappings": { # describes how fields in the database documents are indexed and stored
            "dynamic": True, # automatically index new fields that appear in the document
            "fields": { # properties of the fields that will be indexed.
                text_embedding_field_name: { 
                    "dimensions": 1536, # size of the vector.
                    "similarity": "cosine", # algorithm used to compute the similarity between vectors
                    "type": "knnVector",
                }
            },
        }
    },
    name=vector_search_index_name_text, # identifier for the vector search index
)


# Check if the index already exists
index_exists = False
for index in collection.list_indexes():
    print(index)
    if index['name'] == vector_search_index_name_text:
        index_exists = True
        break


# Create the index if it doesn't exist
if not index_exists:
    try:
        result = collection.create_search_index(model=vector_search_index_model)
        print("Creating index...")
        time.sleep(20)  # Sleep for 20 seconds, adding sleep to ensure vector index has compeleted inital sync before utilization
        print("Index created successfully:", result)
        print("Wait a few minutes before conducting search with index to ensure index intialization")
    except Exception as e:
        print(f"Error creating vector search index: {str(e)}")
else:
    print(f"Index '{vector_search_index_name_text}' already exists.")



openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small", 
            dimensions=1536
        ).data[0].embedding

        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None
    

def vector_search(user_query, db, collection, vector_index="vector_index"):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    db (MongoClient.database): The database object.
    collection (MongoCollection): The MongoDB collection to search.
    additional_stages (list): Additional aggregation stages to include in the pipeline.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search stage
    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_index, # specifies the index to use for the search
            "queryVector": query_embedding, # the vector representing the query
            "path": text_embedding_field_name, # field in the documents containing the vectors to search against
            "numCandidates": 150, # number of candidate matches to consider
            "limit": 20 # return top 20 matches
        }
    }

    pipeline = [vector_search_stage]

    results = collection.aggregate(pipeline)

    return list(results)



def handle_user_query(query, db, collection):
    # Assuming vector_search returns a list of dictionaries with keys 'title' and 'plot'
    get_knowledge = vector_search(query, db, collection)

    if not get_knowledge:
        return "No results found.", "No source information available."
        
     # Convert search results into a list of SearchResultItem models
    search_results_models = [
        SearchResultItem(**result)
        for result in get_knowledge
    ]

    # Convert search results into a DataFrame for better rendering in Jupyter
    search_results_df = pd.DataFrame([item.dict() for item in search_results_models])

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a airbnb listing recommendation system."},
            {
                "role": "user", 
                "content": f"Answer this user query: {query} with the following context:\n{search_results_df}"
            }
        ]
    )

    system_response = completion.choices[0].message.content

    print(f"- User Question:\n{query}\n")
    print(f"- System Response:\n{system_response}\n")

    display(HTML(search_results_df.to_html()))

    return system_response


# query = """
# I want to stay in a place that's warm and friendly, 
# and not too far from resturants, can you recommend a place? 
# Include a reason as to why you've chosen your selection.
# """
query = """
I want to stay in a place that is a fantastic duplex apartment with three bedrooms. 
The place should have a view of a river. 
There should be a metro station for transport.
"""
handle_user_query(query, db, collection)




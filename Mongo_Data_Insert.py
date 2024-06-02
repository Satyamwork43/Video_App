import json
from pymongo import MongoClient

# File path to the JSON file
json_file_path = '/content/drive/MyDrive/output_data.json'

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    output_data = json.load(json_file)

# MongoDB connection details
client = MongoClient(
    "my_uri",
    tls=True,
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=30000
)

# Database and collection name
database_name = "satyam"
collection_name = "ashu"
db = client[database_name]
collection = db[collection_name]

# Insert data into MongoDB
if isinstance(output_data, list):
    collection.insert_many(output_data)
else:
    collection.insert_one(output_data)

print("Data inserted successfully!")

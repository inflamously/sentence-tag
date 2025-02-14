import json
import os.path

import tqdm

from db import PromptDB
from utils import split_into_chunks

documents = []

# --- Load inputs
if os.path.exists('./inputs.txt'):
    with open('./inputs.txt', 'r', encoding="utf-8") as f:
        documents = f.readlines()
else:
    raise FileNotFoundError('./inputs.txt')

if len(documents) == 0:
    raise Exception('There are no input sample prompts')

# --- Load model
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")

# --- Load entities to add to db
entities = None

if os.path.exists('./entities.txt'):
    with open('./entities.txt', 'r', encoding="utf-8") as f:
        entities = f.readlines()
else:
    raise FileNotFoundError('./entities.txt')

# --- Initialize Database
db = PromptDB()

for entity in entities:
    db.add_entity(entity.replace("\n", "").replace("\r", ""))
labels = db.get_entities()

# --- Inference
CHUNK_SIZE = 32

for chunked_documents in tqdm.tqdm(
        split_into_chunks(documents, CHUNK_SIZE) if len(documents) >= CHUNK_SIZE else documents):
    # for document in tqdm.tqdm(documents):
    list_of_entities = model.batch_predict_entities(chunked_documents, labels, threshold=0.1)
    for entity_list_index in range(len(list_of_entities)):
        document = chunked_documents[entity_list_index].replace("\n", "").replace("\r", "")
        if len(list_of_entities[entity_list_index]) == 0:
            print("no entities for document {} found.".format(document))
            db.add_unlabeled_prompt(document)
        for entity in list_of_entities[entity_list_index]:
            try:
                db.add_prompt(entity["label"], document)
            except Exception as e:
                print(e)

db.close()

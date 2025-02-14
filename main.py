import json
import os.path

import tqdm

documents = []

if os.path.exists('./inputs.txt'):
    with open('./inputs.txt', 'r', encoding="utf-8") as f:
        documents = f.readlines()
else:
    raise FileNotFoundError('./inputs.txt')

if len(documents) == 0:
    raise Exception('There are no input sample prompts')

from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")

PROMPT_MAP: dict

if os.path.exists("./prompt_map.json"):
    with open("./prompt_map.json", "r", encoding="utf-8") as f:
        PROMPT_MAP = json.loads(f.read())
else:
    PROMPT_MAP = {
        "#": 0,
        "#_UNLABELED_#": [],
        "Person": [],
        "Animal": [],
        "Building/Architecture": [],
        "Vehicle/Transportation": [],
        "Landscape/Nature": [],
        "Urban/Cityscape": [],
        "Event/Occasion": [],
        "Technology": [],
        "Food & Drink": [],
        "Logo/Brand": [],
        "Fashion/Clothing": [],
        "Art & Design": [],
        "Sports & Recreation": [],
        "Medical/Health": [],
        "Scientific/Research": [],
        "Historical": [],
        "Political/Government": [],
        "Economic/Business": [],
        "Cultural/Heritage": [],
        "Abstract/Conceptual": [],
        "Landscape": [],
        "Cityscape": [],
        "Underwater": [],
        "Aerial/Drone": [],
        "Night Scene": [],
        "Macro": [],
        "Black & White": [],
        "Digital Art": [],
        "Illustration": [],
        "3D Render": [],
        "Wildlife": [],
        "Flora/Botanical": [],
        "Desert": [],
        "Forest": [],
        "Mountain": [],
        "Beach/Coastal": [],
        "River/Lake": [],
        "Snow/Ice": [],
        "Sunrise/Sunset": [],
        "Architecture Detail": [],
        "Interior Design": [],
        "Street Photography": [],
        "Festival/Celebration": [],
        "Holiday": [],
        "Portrait": [],
        "Group Photo": [],
        "Selfie": [],
        "Action/Sports": [],
        "Dance": [],
        "Music/Concert": [],
        "Theater/Performance": [],
        "Exhibition/Show": [],
        "Cartoon/Animation": [],
        "Graffiti/Street Art": [],
        "Wildlife Close-Up": [],
        "Insect Macro": [],
        "Bird Photography": [],
        "Aquatic Life": [],
        "Reptile/Amphibian": [],
        "Pet Photography": [],
        "Wild Landscape": [],
        "Panorama": [],
        "Seasonal": [],
        "Weather": [],
        "Cloudscape": [],
        "Space/Astronomy": [],
        "Galaxy/Starfield": [],
        "Moon/Lunar": [],
        "Satellite Imagery": [],
        "Microscopic": [],
        "X-ray/Medical Imaging": [],
        "Diagram/Infographic": [],
        "Cartography/Maps": [],
        "Scientific Illustration": [],
        "Architectural Blueprint": [],
        "Vintage/Retro": [],
        "Minimalist": [],
        "Surreal": [],
        "Fantasy": [],
        "Sci-Fi": [],
        "Horror/Gothic": [],
        "Cinematic": [],
        "Documentary": [],
        "Photojournalism": [],
        "Wildlife Documentary": [],
        "Travel": [],
        "Adventure": [],
        "Outdoors": [],
        "Rural": [],
        "Suburban": [],
        "Industrial": [],
        "Construction": [],
        "Agriculture/Farming": [],
        "Energy/Power": [],
        "Environmental Conservation": [],
        "Culinary Art": [],
        "Beverage": [],
        "Still Life": [],
        "Abstract Pattern": [],
        "Mixed Media": [],
        "Decoration": [],
    }

labels = list(filter(lambda key: key != '#' and key != '#_UNLABELED_#', PROMPT_MAP.keys()))


def has_prompt(prompt):
    for key in PROMPT_MAP.keys():
        if key == '#': continue
        if prompt in PROMPT_MAP[key]:
            return True
    return False


def split_into_chunks(lst, chunk_size=4):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


CHUNK_SIZE = 32

for chunked_documents in tqdm.tqdm(split_into_chunks(documents, CHUNK_SIZE) if len(documents) >= CHUNK_SIZE else documents):
    # for document in tqdm.tqdm(documents):
    list_of_entities = model.batch_predict_entities(chunked_documents, labels, threshold=0.1)
    for entity_list_index in range(len(list_of_entities)):
        document = chunked_documents[entity_list_index]
        if len(list_of_entities[entity_list_index]) == 0:
            print("no entities for document {} found.".format(document))
            PROMPT_MAP['#_UNLABELED_#'].append(document)
        for entity in list_of_entities[entity_list_index]:
            if not has_prompt(document):
                PROMPT_MAP[entity["label"]].append(document)
                PROMPT_MAP['#'] += 1

with open("./prompt_map.json", "w") as f:
    f.write(json.dumps(PROMPT_MAP, indent=4))

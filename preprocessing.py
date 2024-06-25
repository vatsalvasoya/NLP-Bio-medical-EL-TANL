import os
import json
import re
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
model = model.to(device)
model.eval()


with open("dataset/processed_data/synonyms.json", "r") as f:
    synonyms = json.load(f)

def compute_similarity(sentence1, sentence2):
    tokens1 = tokenizer.encode(sentence1, add_special_tokens=True)
    tokens2 = tokenizer.encode(sentence2, add_special_tokens=True)

    with torch.no_grad():
        outputs1 = model(torch.tensor([tokens1]).to(device))
        outputs2 = model(torch.tensor([tokens2]).to(device))

    # Get the CLS token embeddings of mention entity and synonym
    embeds1 = outputs1.last_hidden_state[:, 0, :].cpu().numpy() 
    embeds2 = outputs2.last_hidden_state[:, 0, :].cpu().numpy()

    similarity = cosine_similarity(embeds1, embeds2)
    
    return similarity[0][0]

def find_synonym(mention_entity, target_id):
    similarity_scores = []

    if target_id in synonyms and len(synonyms[target_id]) > 0: # see if target_id is present in the synonyms.json
        for syn in synonyms[target_id]: # For all synonyms compute similarity scores
            similarity_scores.append(compute_similarity(mention_entity, syn)) 

        pairs = list(zip(synonyms[target_id], similarity_scores))
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True) # Sort the synonyms in desc order of similarity scores.

        return sorted_pairs[0][0]
    
    return None

def preprocess_data(raw_data_dir, processed_data_dir, raw_data_file, processed_data_file, entity_file):
    print(f"Processing {raw_data_file}")
    
    regex = re.compile('^\d+\|[a|t]\|')
    
    with open(os.path.join(processed_data_dir, entity_file), "r") as f:
        entities = json.load(f)

    documents = dict()
    annotated_docs = []
    start_token = "["
    end_token = "]"

    with open(os.path.join(raw_data_dir, raw_data_file), encoding='utf-8') as f:
        syn_not_found = 0
        for line in f:
            doc = dict()
            line = line.strip()
            if regex.match(line):
                match_span = regex.match(line).span()
                start_span_idx = match_span[0]
                end_span_idx = match_span[1]

                document_id = line[start_span_idx:end_span_idx].split("|")[0]
                text = line[end_span_idx:]

                if document_id not in documents:
                    documents[document_id] = text
                else:
                    documents[document_id] = documents[document_id] + ' ' + text

            else:
                cols = line.strip().split('\t')
                if len(cols) == 6:
                    if cols[5] == '-1':
                        continue
                    document_id = cols[0]
                    
                    start_index = int(cols[1])
                    end_index = int(cols[2])
                    ids = cols[5].split("|")
                    entity_type = cols[4]

                    for entity_id in ids:
                        if entity_id in entities:
                            input_text = documents[document_id]
                            
                            # NOTE: Use entity_type_str if you want to incorporate the entity type in input and output.
                            # entity_type_str = " | type = " + entity_type
                            mention_entity = input_text[start_index:end_index]
                            synonym = None

                            # NOTE: uncomment this line if you want to add synonyms in the output
                            # synonym = find_synonym(mention_entity, entity_id) 


                            input_text = input_text[:end_index] + end_token + input_text[end_index:]
                            input_text = input_text[:start_index] + start_token + input_text[start_index:]

                            output_text = documents[document_id]
                            if synonym:
                                target_entity_str = " | target = " + entities[entity_id] + " | synonym = " + synonym
                            else:
                                syn_not_found += 1
                                target_entity_str = " | target = " + entities[entity_id]
                            output_text = output_text[:end_index] + target_entity_str + end_token + output_text[end_index:]
                            output_text = output_text[:start_index] + start_token + output_text[start_index:]

                            doc["input"] = input_text

                            # No need to add output string in the output
                            if raw_data_file != "test_corpus.txt":
                                doc["output"] = output_text
                            doc["target"] = entities[entity_id]
                            doc["target_id"] = entity_id
                            # doc["entity_type"] = entity_type_str

                            annotated_docs.append(doc)
                        else:
                            print(f"{entity_id} not found in entities.")


    
    with open(os.path.join(processed_data_dir, processed_data_file), 'w') as f:
        for item in annotated_docs:
            f.write(json.dumps(item) + "\n")
                    
    print("syn_not_found: ", syn_not_found)

if __name__ == "__main__":
    files = ["train_corpus", "dev_corpus", "test_corpus"] # List of files you need to preprocess from the raw data directory

    for fname in files:
        preprocess_data("dataset/raw_data/", "dataset/processed_data/", f"{fname}.txt", f"{fname}.jsonl", "old_entities.json")


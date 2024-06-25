from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from torch.utils.data import Dataset, DataLoader
import json
from trie import Trie
from tqdm import tqdm
import re

# Regex to decode the synonym and target from the output.
target_pattern = re.compile(r'\[[^\]]*?target = (.*?)(?=\|)')
synonym_pattern = re.compile(r'\[[^\]]*?synonym = (.*?)\]')


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-3b")
batch_size = 8

# Qlora config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


class EntityDisambiguationDataset(Dataset):
    def __init__(self, input_text, target_labels, target_id):
        self.input_text = input_text
        self.target_labels = target_labels
        self.target_ids = target_ids

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        input_text = self.input_text[idx]
        target_label = self.target_labels[idx]
        target_id = self.target_ids[idx]

        
        input_encoded = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        input_ids = input_encoded["input_ids"].squeeze()

        return {
            "input_ids": (input_ids).to(device),
            "target_label": target_label,
            "target_ids": target_id
        }


input_text = []
target_labels = []
target_ids = []
with open("dataset/processed_data/test_corpus.jsonl") as f:
    for line in f:
        data = json.loads(line)
        input_text.append(data["input"])
        target_labels.append(data["target"])
        target_ids.append(data["target_id"])

dataset = EntityDisambiguationDataset(input_text, target_labels, target_ids)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


with open("dataset/processed_data/synonyms.json") as f:
    synonyms = json.load(f)


total_samples = len(input_text)
for checkpoint in range(60, 61, 1): # Number of epochs you need inference for
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(
                'models/{}/'.format(checkpoint), 
                quantization_config=bnb_config)
    model = model.eval().to(device)

    correct_samples = 0
    num = 0
    target_count = 0
    syn_count = 0
    total_syn_count = 0
    predictions = []
    for batch in tqdm(dataloader, desc="Processing "):
        input_ids = batch["input_ids"]
        target_labels = batch["target_label"]
        target_ids = batch["target_ids"]

        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                decoder_start_token_id=tokenizer.bos_token_id,
                max_length=512
            )

        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for output, target, target_id in zip(decoded_outputs, target_labels, target_ids):
            num += 1
            predicted_target = target_pattern.search(output) # get the predicted target
            predicted_synonym = synonym_pattern.search(output) # get the predicted synonym

            # print(f"target: {target}, output: {output}")

            # print(f"target: {target}")
            # if predicted_target:
            #     print(f"predicted_target: {predicted_target.group(1).strip()}")
            # if predicted_synonym:
            #     print(f"predicted_synonym: {predicted_synonym.group(1).strip()}")


            # heuristics to check if model was able to predict either the gold entity or the synonym or none of them. 
            if predicted_target and (predicted_target.group(1).strip() == target):
                correct_samples += 1
                target_count += 1
                if predicted_synonym and (target_id in synonyms) and (predicted_synonym.group(1).strip() in synonyms[target_id]):
                    total_syn_count += 1
            elif predicted_synonym and (target_id in synonyms) and (predicted_synonym.group(1).strip() in synonyms[target_id]):
                correct_samples += 1
                syn_count += 1
                total_syn_count += 1
            
            #     print(f"{num} | predicted_target: {predicted_target} | target: {target} | correct count: {correct_samples}")
            # else:
            #     print(f"{num} | target not found. | output = {output} | target: {target}")


            predictions.append({
                "output": output,
                "target": target,
                "target_id": target_id
            })


    print(f"Checkpoint: {checkpoint}, Accuracy: {correct_samples/total_samples}")
    print(f"target count: {target_count}, synonym count: {syn_count}, total synonym count: {total_syn_count}")

    # Dumping the predictions in a jsonl file
    with open(f"predictions/{checkpoint}.jsonl", 'w') as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")





    # correct_samples = 0
    # num = 0
    # predictions = []
    # for input, target, target_id in tqdm(zip(input_text, target_labels, target_ids), desc="Processing "):
    #     # num += 1
    #     input_ids = (tokenizer.encode(input, return_tensors="pt", add_special_tokens=False)).to(device)

    #     outputs = model.generate(
    #         inputs=input_ids,
    #         decoder_start_token_id=tokenizer.bos_token_id,
    #         max_length=1024
    #     )

    #     output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     predictions.append({
    #         "output": output,
    #         "target": target,
    #         "target_id": target_id
    #     })

    # with open(f"predictions/{checkpoint}.jsonl", 'w') as f:
    #     for item in predictions:
    #         f.write(json.dumps(item) + "\n")

        # predicted = pattern.search(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # if predicted:
        #     predicted = predicted.group(1).strip()
        #     if predicted == target:
        #         correct_samples += 1
        #     print(f"{num} | predicted: {predicted} | target: {target} | correct count: {correct_samples}")
        # else:
        #     print(f"{num} | target not found. | output = {tokenizer.decode(outputs[0], skip_special_tokens=True)} | target: {target}")
    # print(f"Checkpoint: {checkpoint}, Accuracy: {correct_samples/total_samples}")

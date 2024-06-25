import re
from tqdm import tqdm
import json

target_pattern = re.compile(r'\[[^\]]*?target = (.*?)(?=\|)')
synonym_pattern = re.compile(r'\[[^\]]*?synonym = (.*?)\]')

with open("dataset/processed_data/synonyms.json") as f:
    synonyms = json.load(f)

# print(len(entities), len(rev_entities))
for checkpoint in range(43, 44, 1):
    outputs = []
    target_labels = []
    target_ids = []
    with open(f"predictions/{checkpoint}.jsonl") as f:
        for line in f:
            data = json.loads(line)
            outputs.append(data['output'])
            target_labels.append(data['target'])
            target_ids.append(data['target_id'])
    

    correct_samples = 0
    target_count = 0
    total_syn_count = 0
    syn_count = 0
    total_samples = len(outputs)
    for output, target, target_id in tqdm(zip(outputs, target_labels, target_ids), desc="Processing "):        
        predicted_target = target_pattern.search(output)
        predicted_synonym = synonym_pattern.search(output)

        if predicted_target and (predicted_target.group(1).strip() == target):
            correct_samples += 1
            target_count += 1
            if predicted_synonym and (target_id in synonyms) and (predicted_synonym.group(1).strip() in synonyms[target_id]):
                total_syn_count += 1
        elif predicted_synonym and (target_id in synonyms) and (predicted_synonym.group(1).strip() in synonyms[target_id]):
            correct_samples += 1
            syn_count += 1
            total_syn_count += 1
        elif predicted_target and (target_id in synonyms) and (predicted_target.group(1).strip() in synonyms[target_id]):
            correct_samples += 1
        elif predicted_synonym and (predicted_synonym.group(1).strip() == target):
            correct_samples += 1
    
    
    print(f"target count: {target_count}, synonym count: {syn_count}, total synonym count: {total_syn_count}")
    print(f"Checkpoint: {checkpoint}, Accuracy: {correct_samples/total_samples}")




































# text = "[Famotidine | type = Chemical | target = Scleroderma, Systemic,]-associated delirium. [Famotidine | type = Chemical | target = Vasoya]A series of six cases. Famotidine is a histamine H2-receptor antagonist used in inpatient settings for prevention of stress ulcers and is showing increasing popularity because of its low cost. Although all of the currently available H2-receptor antagonists have shown the propensity to cause delirium, only two previously reported cases have been associated with famotidine. The authors report on six cases of famotidine-associated delirium in hospitalized patients who cleared completely upon removal of famotidine. The pharmacokinetics of famotidine are reviewed, with no change in its metabolism in the elderly population seen. The implications of using famotidine in elderly persons are discussed."

# target = re.search(r'\[[^\]]*?target = (.*?)\]', text).group(1).strip()
# # target = re.search(r'target = (\[.+?\])', text).group(1).strip()
# # print(target.strip())
# print(target)


# match = re.search(r'\[(.*?)\]', text).group(1)
# index = match.find("target = ")
# if index != -1:
#     index += len("target = ")
#     print(match[index:])




# # First WAY
# predicted = re.search(r'target = ([^\[\]]+)', tokenizer.decode(outputs[0], skip_special_tokens=True))
# if predicted:
#     predicted = predicted.group(1).strip()

#     if predicted == target:
#         correct_samples += 1


# # Second WAY
# predicted = re.search(r'\[(.*?)\]', tokenizer.decode(outputs[0], skip_special_tokens=True))
# if predicted:
#     predicted = predicted.group(1)
#     index = predicted.find("target = ")
#     if index != -1:
#         index += len("target = ")
#         if predicted[index:] == target:
#             correct_samples += 1


# Third WAY
# pattern = re.compile(r'\[[^\]]*?target = (.*?)\]')
# predicted = pattern.search(tokenizer.decode(outputs[0], skip_special_tokens=True))
# if predicted:
#     predicted = predicted.group(1).strip()
#     if predicted == target:
#         correct_samples += 1

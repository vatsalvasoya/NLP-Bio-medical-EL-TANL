import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AdamW
from tqdm import tqdm
import os


batch_size = 8
num_epochs = 500
teacher_forcing_epochs = 0   # No of epochs for which you want to apply teacher forcing
learning_rate = 1e-4
checkpoint_rate = 1  # Interval to save the checkpoints of the model during training.
train_val_split_ratio = 0.9  # train to validation set split ratio. 0.9 means 90% training and 10% validaton
train_dataset_name = "dataset/processed_data/train_dev_corpus.jsonl"


# Setting CUDA device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

 
# Configurations for the Qlora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-3b")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b", quantization_config=bnb_config)


# Setup for the Qlora. If you don't want to use Qlora comment these two lines and remove "quantization_config" from above.
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Move the model to CUDA
model.to(device)



# Creating dataset class
class EntityDisambiguationDataset(Dataset):
    def __init__(self, input_text, output_text, tokenizer):
        self.input_text = input_text
        self.output_text = output_text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        input_text = self.input_text[idx]
        output_text = self.output_text[idx]

        # KEEPING THE SPECIAL TOKENS
        input_encoded = self.tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        input_ids = input_encoded["input_ids"].squeeze()
        attention_mask = input_encoded["attention_mask"].squeeze()

        output_encoded = self.tokenizer(output_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        output_ids = output_encoded["input_ids"].squeeze()

        return {
            "input_ids": (input_ids).to(device),
            "attention_mask": (attention_mask).to(device),
            "output_ids": (output_ids).to(device)
        }


# Creating the dataloader
def create_dataloader(dataset_name):
    input_text = []
    output_text = []
    with open(dataset_name) as f:
        for line in f:
            data = json.loads(line)
            input_text.append(data["input"])
            output_text.append(data["output"])

    dataset = EntityDisambiguationDataset(input_text, output_text, tokenizer)

    train_size = int(train_val_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

# Create train and validation loader
train_dataloader, val_dataloader = create_dataloader(train_dataset_name)


# Loss info you need to store in loss.json
loss_info = {
    "epoch": list(),
    "tr_loss": list(),
    "val_loss": list()
}

# Training the model
optimizer = AdamW(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    teacher_forcing_flag = (epoch < teacher_forcing_epochs) # Check if teacher forcing needs to be applied or not.

    # Training
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        output_ids = batch["output_ids"]

        optimizer.zero_grad()
        if teacher_forcing_flag:
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids, decoder_input_ids=output_ids).loss
        else:
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids).loss

        epoch_loss += loss.item()

        loss.backward()  
        optimizer.step()

    average_epoch_loss = epoch_loss / len(train_dataloader)

    
    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            val_input_ids = val_batch["input_ids"]
            val_attention_mask = val_batch["attention_mask"]
            val_output_ids = val_batch["output_ids"]

            val_loss += model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_output_ids).loss.item()

    average_val_loss = val_loss / max(1, len(val_dataloader))
    print("Training loss: {}, Validation Loss: {}".format(average_epoch_loss, average_val_loss))

    loss_info["epoch"].append(epoch)
    loss_info["tr_loss"].append(average_epoch_loss)
    loss_info["val_loss"].append(average_val_loss)


    # if ((epoch+1)%checkpoint_rate == 0) and (epoch+1) > teacher_forcing_epochs: 
    if ((epoch+1)%checkpoint_rate == 0):
        os.mkdir(os.path.join("models/", "{}".format(epoch+1)))
        model.save_pretrained("models/{}".format(epoch+1))

with open("loss.json", 'w') as f:
    json.dump(loss_info, f)

# NLP-Bio-medical-EL-TANL
A Sequence-to-Sequence model for Bio-medical Entity Normalization

### Steps to run experiments without QLORA:
    1. Create and activate virtual env in conda: 
        - conda create -n vatsal-env python=3.7
        - conda activate vatsal-env
    2. Install dependency:
        - pip install torch torchvision
        - pip install transformers
        - pip install sentencepiece // FOR T5 Models only

### Steps to run experiments with QLORA:
    1. Create virtual env:
        - conda create --name vatsal-env-qlora python=3.10 -y
        - source activate vatsal-env-qlora

    2. Installing dependencies:
        - pip install peft
        - pip install bitsandbytes
        - pip install torch==2.0.0
        - pip install sentencepiece // FOR T5 Models only

### Run model and inference files
    1. Run model.py
        - Run model.py in the background: nohup python model.py > output.log 2>&1 &
        - To see the logs: tail -f output.log
        
    2. Run infernece.py
        - nohup python inference.py > output_inference.log 2>&1 &
	

### Detailed information on directory structure and files:
    1. preprocessing.py
        - To process the train, dev and test data from "dataset/raw_data" and processed data will be stored in "dataset/processed_data".
        
    2. model.py
        - It generates "output.log" and "loss.json" during execution.
        - It stores the model checkpoints in "models/" directory.
    3. inference.py
        - It stores the predictions of TANL formulation on test set in "predictions/" directory.
        - "predictions/" directory is created so that you don't have to do the inference for the same checkpoint again.

    4. postprocessing.py
        - only used in TANL formulation.
        - It takes a file from "predictions/" directory and postprocess it.
        - you can use this file to do postprocessing on the output generated during the inference.
        
    5. parsing.py (Only in END TO END EL)
        - to parse the output and get all the entity recognized by the model alongside the target entity.
        - the "parse_output_sentence" function is taken from the repo of TANL paper: https://github.com/amazon-science/tanl/blob/fdcf4f8bc4b63b3551da23312e332e5d3c413a86/output_formats.py
        - It will be usedful in END TO END EL for the evaluation after the inference is done.
        
    6. augmentation.py (It will be present in the directory if only it is required)
        - It is used to choose synonyms based on cosine similarity scores computed from [CLS] embeddings.(Check the report/PPT to understand what I mean here)

    7. trie.py (It will be present in the directory if only it is required)
        - only used in GENRE formulation and it was taken from the repo of that paper.
        - NO NEED TO READ THE CODE. Just see how it is used in "inference.py" to build a trie and to add entity name in the trie.
        
    8. loss.py (It will be present in the directory if only it is required)
        - It is used to parse the "output.log" and get the train and validation loss values from it.
        - It can be useful if you stopped execution of "model.py" in between and "loss.json" file is not stored then you can get the loss values from "output.log"
        
    9. dataset/processed_data/old_entities.json or entities.json
        - It contains ID to name mapping of all gold entities.
        - It is created from "entities.txt" files from "dataset/raw_data".
        - USE ONLY "old_entities.json" and "old_entities.json". USE entities file only if it contains 29055 entries. DON'T USE the entities file containing 29383 entries.
        
    10. dataset/processed_data/synonyms.json
        - It contains all synonyms for the gold entities.
        
    11. dataset/processed_data/train_dev_corpus.json
        - It contains combined data of dev and train corpus.

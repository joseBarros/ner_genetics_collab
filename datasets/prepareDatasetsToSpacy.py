from datasets import load_dataset

# https://huggingface.co/datasets/jnlpba
train_jnlpba_Dataset = load_dataset("jnlpba", split="train")
dev_jnlpba_Dataset = load_dataset("jnlpba", split="validation")

#https://huggingface.co/datasets/bc2gm_corpus
train_BC2GM_Dataset = load_dataset("bc2gm_corpus", split="train")
dev_BC2GM_Dataset = load_dataset("bc2gm_corpus", split="validation")

#https://huggingface.co/datasets/ncbi_disease
train_NCBI_disease_Dataset = load_dataset("ncbi_disease", split="train")
dev_NCBI_disease_Dataset = load_dataset("ncbi_disease", split="validation")

## GET LABELS FROM DATASET -> https://huggingface.co/docs/transformers/tasks/token_classification
tags_jnlpba = train_jnlpba_Dataset.features[f"ner_tags"].feature.names
print(tags_jnlpba)
tags_BC2GM = train_BC2GM_Dataset.features[f"ner_tags"].feature.names
print(tags_BC2GM)
tags_NCBI = train_NCBI_disease_Dataset.features[f"ner_tags"].feature.names
print(tags_NCBI)
print("######")
#tags = tags_jnlpba + tags_BC2GM[1:] + tags_NCBI[1:]
#print(tags)

lenJNLPBA = len(tags_jnlpba)
lenBC2GM = len(tags_BC2GM)
lenNCBI = len(tags_NCBI)

print(lenJNLPBA)
print(lenBC2GM)
print(lenNCBI)
print("####")

#print(train_BC2GM_Dataset[:1])

# Replacing 1 with 11 and 2 with 20 in ner_tags
# for i, tags in enumerate(train_BC2GM_Dataset['ner_tags':1]):
#     train_BC2GM_Dataset['ner_tags'][i] = [lenJNLPBA+1 if tag == 1 else (lenJNLPBA+1 if tag == 2 else tag) for tag in tags]
#
# print(train_BC2GM_Dataset[:1])

def tokens_to_sentence(tokens):
    # Initialize an empty list to store the processed tokens
    processed_tokens = []

    # Initialize a variable to keep track of open parentheses
    open_parentheses = False

    # import string library function
    import string
    #!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    # List of special characters
    special_characters = list(string.punctuation) + ["'s"]

    # Iterate over each token
    for i, token in enumerate(tokens):
        # Check if the token is a special character
        if token in special_characters:
            # If it is, handle it based on its type
            if token == ".":
                if i > 0:
                    processed_tokens[-1] += "."
            elif token == "'s" and i > 0:
                processed_tokens[-1] += "'s"
            elif token == "(":
                open_parentheses = True
                processed_tokens.append(token)
            elif token == ")":
                open_parentheses = False
                if i > 0:
                    processed_tokens[-1] += ")"
            else:
                # Add the special character to the processed tokens
                processed_tokens.append(token)
        else:
            # If the token is not a special character, handle it accordingly
            if open_parentheses and i > 0:
                processed_tokens[-1] += token
            else:
                processed_tokens.append(token)

    # Join the processed tokens into a sentence
    sentence = " ".join(processed_tokens)
    return sentence

def find_word_positions(sentence, word):
    tokens = sentence.split()
    start_pos = 0
    for token in tokens:
        end_pos = start_pos + len(token)
        if token == word:
            return start_pos, end_pos
        start_pos = end_pos + 1  # Add 1 for space between words
    return None, None  # Word not found


import re


def find_word_positions_regex(text, word):
    positions = []
    pattern = re.compile(r'\b{}\b'.format(re.escape(word)))
    for match in re.finditer(pattern, text):
        positions.append((match.start(), match.end()))
    return positions


def cut(limit, data, tags):
    i = 0
    k = 0

    #temp = [] if training_data is None else training_data
    temp = []

    for line in data:
        frase = " ".join(line['tokens'])
        #frase = tokens_to_sentence(line['tokens'])

        if(frase in frases):
            continue

        result = []
        i = 0
        lastPos = 0
        for num in line['ner_tags']:
            result.append(tags[num])
            word = line['tokens'][i]

            start_pos, end_pos = find_word_positions(frase[lastPos:], word)
            # print(end_pos)
            print(frase[lastPos:])
            if(lastPos is None or start_pos is None or end_pos is None):
                continue
            temp.append((frase, [(lastPos + start_pos, lastPos + end_pos, tags[num])]))
            print(lastPos + start_pos, lastPos + end_pos)
            lastPos = lastPos + end_pos + 1
            i = i + 1

        frases.append(frase)
        # if (k == limit):
        #     break
        k = k + 1
        i = 0

    return temp

numTrain=5
frases = []
training_jnlpba = cut( numTrain, train_jnlpba_Dataset, tags_jnlpba)
training_BC2GM = cut( numTrain, train_BC2GM_Dataset, tags_BC2GM)
training_NCBI = cut( numTrain, train_NCBI_disease_Dataset, tags_NCBI)
#training_data = training_jnlpba + training_BC2GM + training_NCBI
training_data = training_jnlpba

numDev=5
dev_jnlpba = cut(numDev, dev_jnlpba_Dataset, tags_jnlpba)
dev_BC2GM = cut(numDev, dev_BC2GM_Dataset, tags_BC2GM)
dev_NCBI = cut(numDev, dev_NCBI_disease_Dataset, tags_NCBI)
#dev_data = dev_jnlpba + dev_BC2GM + dev_NCBI
dev_data = dev_jnlpba
print(training_data)
print(dev_data)

import spacy
from spacy.tokens import DocBin
# the DocBin will store the example documents
db = DocBin()
nlp = spacy.blank("en")

def textToSpacy(nameSpacyFile, data):
    db = DocBin()
    textos = []
    for text, annotations in data:

        if (text == "( ABSTRACT TRUNCATED AT 250 WORDS )"):
            print("ABSTRACT TRUNCATED AT 250 WORDS")
            continue

        if (text == "RESULTS ."):
            print("RESULTS")
            continue
        if(text == "OBJECTIVE ."):
            print("OBJECTIVE")
            continue
        if(text == "METHODS ."):
            print("METHODS")
            continue

        if(text == "CONCLUSION ."):
            print("CONCLUSION")
            continue

        # if (text in textos):
        #     continue

        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)


        doc.ents = ents
        db.add(doc)

        textos.append(text)
    db.to_disk("../corpus/" + nameSpacyFile + ".spacy")
    return db

def toJsonSpacy(name, docbin):
    import json
    # Create a list to store the converted documents
    converted_docs = []

    # Create a dictionary to store the converted documents
    converted_docs_dict = {}
    # Iterate over the DocBin and extract relevant information
    for doc in docbin.get_docs(nlp.vocab):
        # Extract entities for the document
        entities = []
        for ent in doc.ents:
            entities.append([ent.start_char, ent.end_char, ent.label_])

        # Check if the document text is already in the dictionary
        if doc.text in converted_docs_dict:
            # Append entities to the existing list
            converted_docs_dict[doc.text]["entities"].extend(entities)
        else:
            # Create a new entry for the document text
            converted_docs_dict[doc.text] = {"entities": entities}

    # Convert the dictionary to a list of tuples
    converted_docs = [[text, data] for text, data in converted_docs_dict.items()]

    # Save the converted documents to a JSON file
    with open("../assets/" + name, "w") as json_file:
        json.dump(converted_docs, json_file, indent=4)


db=textToSpacy("train", training_data)
toJsonSpacy("train.json", db)

db=textToSpacy("dev", dev_data)
toJsonSpacy("dev.json", db)

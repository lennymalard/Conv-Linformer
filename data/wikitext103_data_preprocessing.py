from datasets import load_dataset
from transformers import RobertaTokenizerFast
import h5py
import re
from random import random, choice
from tqdm import tqdm
import sys

SEQ_LENGTHS = [128, 256, 512, 1024]
DATASET_TOKEN_SIZE = 25_000_000
PATH = "./pretraining/wikitext103"

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train",  streaming=True)
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

CLS_ID = tokenizer.cls_token_id
PAD_ID = tokenizer.pad_token_id
SEP_ID = tokenizer.sep_token_id
MASK_ID = tokenizer.mask_token_id

def text_cleaning(text):
    text = re.sub(r'=+\s.*\s+', '', text)
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r'@(.)@', r'\1', text)
    text = re.sub(r'\s+([.?!,;\'])', r'\1', text)
    text = re.sub(r'\s+-\s+', '-', text)
    text = re.sub(r'([\[{("\'])\s+', r'\1', text)
    text = re.sub(r'\s+([]})"\'])', r'\1', text)
    text = re.sub(r'\s+', r' ', text)
    return text

def text_normalization(text):
    return text.lower()

def mlm_masking(ids, tokenizer, mask_prob=0.15):
    ids = ids.copy()
    mask = [0] * len(ids)
    labels = [-100] * len(ids)
    vocab_ids = list(tokenizer.get_vocab().values())
    for i in range(1, len(ids)-1):
        if ids[i] == PAD_ID:
            continue
        if random() < mask_prob:
            prob = random()
            if prob <= 0.8:
                labels[i] = ids[i]
                ids[i] = MASK_ID
            elif 0.8 < prob <= 0.9:
                ids[i] = choice(vocab_ids)
            mask[i] = 1
    return ids, mask, labels

def preprocessing_pipeline():
    for seq_length in SEQ_LENGTHS:
        token_count = 0
        buffer = []
        print(f"\nStarting preprocessing for the {seq_length} sequence length dataset...")
        with h5py.File(f"{PATH}wikitext2_{seq_length}_validation.hdf5", "w") as f:
            for i, text in enumerate(tqdm(dataset, file=sys.stdout)):
                text = text['text']
                text = text_cleaning(text)
                text = text_normalization(text)
                if not text or text.isspace():
                    continue
                ids = tokenizer.encode(text, add_special_tokens=True)
                buffer.extend(ids)
                token_count += len(ids)
                if len(buffer) > seq_length - 2:
                    while len(buffer) > seq_length:
                        masked_ids, mask, labels = mlm_masking(buffer[:seq_length - 1] + [SEP_ID], tokenizer)
                        if 'data' in f and 'mask' in f and 'labels' in f:
                            data = f['data'][:]
                            new_shape = (data.shape[0] + 1, data.shape[1])
                            f['data'].resize(new_shape)
                            f['mask'].resize(new_shape)
                            f['labels'].resize(new_shape)
                            f['data'][-1, :] = masked_ids
                            f['mask'][-1, :] = mask
                            f['labels'][-1, :] = labels
                        else:
                            f.create_dataset('data', data=[masked_ids], maxshape=(None, seq_length))
                            f.create_dataset('mask', data=[mask], maxshape=(None, seq_length))
                            f.create_dataset('labels', data=[labels], maxshape=(None, seq_length))
                        buffer = [CLS_ID] + buffer[seq_length - 1:]
                if token_count > DATASET_TOKEN_SIZE:
                    break
        print(f"\nFinished preprocessing for the {seq_length} sequence length dataset.")
        print(f"\nTotal tokens: {token_count}")

if __name__ == "__main__":
    preprocessing_pipeline()
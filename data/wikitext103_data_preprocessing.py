from datasets import load_dataset
from transformers import RobertaTokenizerFast
import h5py
import re
from random import random, choice
from tqdm import tqdm
import sys
import multiprocessing

SEQ_LENGTHS = [128, 256, 512, 1024]
BATCH_SIZE = 4096
DATASET_TOKEN_SIZE = 2_000_000
NUM_WORKERS = 6
PATH = "./pretraining/wikitext103/"

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
                labels[i] = ids[i]
                ids[i] = choice(vocab_ids)
    return ids, labels

def batch_preprocessing(text_batch):
    ids = []
    for text in text_batch:
        text = text_cleaning(text)
        text = text_normalization(text)
        if not text or text.isspace():
            continue
        ids.append(tokenizer.encode(text, add_special_tokens=False))
    return ids

def preprocessing_pipeline():
    global NUM_WORKERS
    for seq_length in SEQ_LENGTHS:
        token_count = 0
        buffer = []
        text_batch = []
        data_batch = []
        label_batch = []
        file_index = 0
        len_data_batch = 0
        print(f"\nStarting preprocessing for the {seq_length} sequence length dataset...")
        with h5py.File(f"{PATH}wikitext103_{seq_length}.hdf5", "w") as f:
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                for i, text in enumerate(tqdm(dataset, file=sys.stdout)):
                    text_batch.append([text['text']])
                    if i % BATCH_SIZE == 0:
                        try:
                            ids_batch = pool.map(batch_preprocessing, text_batch)
                        except Exception as e:
                            print("Multiprocessing error:", e)
                            pool.close()
                            pool.join()
                            continue
                        for ids in ids_batch:
                            if ids:
                                ids = ids[0]
                            buffer.extend(ids)
                            token_count += len(ids)
                            if len(buffer) >= seq_length - 2:
                                while len(buffer) >= seq_length - 2:
                                    data, labels = mlm_masking([CLS_ID] + buffer[:seq_length - 2] + [SEP_ID], tokenizer)
                                    data_batch.append(data)
                                    label_batch.append(labels)
                                    buffer = buffer[seq_length - 2:]
                                len_data_batch = len(data_batch)
                                if 'data' in f and 'labels' in f:
                                    new_shape = (file_index + len_data_batch, seq_length)
                                    f['data'].resize(new_shape)
                                    f['labels'].resize(new_shape)
                                    f['data'][-len(data_batch):, :] = data_batch
                                    f['labels'][-len(label_batch):, :] = label_batch
                                else:
                                    f.create_dataset('data', data=data_batch, maxshape=(None, seq_length))
                                    f.create_dataset('labels', data=label_batch, maxshape=(None, seq_length))
                                data_batch = []
                                label_batch = []
                                file_index += len_data_batch
                        text_batch = []
                    if token_count > DATASET_TOKEN_SIZE:
                            break
        print(f"\nFinished preprocessing for the {seq_length} sequence length dataset.")
        print(f"\nTotal tokens: {token_count}")

if __name__ == "__main__":
    preprocessing_pipeline()
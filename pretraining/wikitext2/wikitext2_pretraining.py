import os

import torch.cuda

from data.utils import HDF5Dataset
from transformers import RobertaTokenizerFast, TrainingArguments, TrainerCallback, Trainer, default_data_collator, \
    TrainerState, TrainerControl, logging
from torch.utils.data import DataLoader
from linformer.linformer import *
from transformer.transformer import *
from tqdm import tqdm
import sys
import numpy as np
from transformers.trainer_callback import PrinterCallback
from math import ceil

SEQ_LENGTH = 256

training_dataset = HDF5Dataset(F'../../data/pretraining/wikitext2/wikitext2_{SEQ_LENGTH}.hdf5')
validation_dataset = HDF5Dataset(F'../../data/pretraining/wikitext2/wikitext2_{SEQ_LENGTH}_validation.hdf5')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

BATCH_SIZE = 8
FULL_BATCH_SIZE = len(training_dataset)

logging.set_verbosity_info()

def preprocess_logits_for_metrics(logits, labels):
    logits = torch.argmax(logits, dim=-1)
    return logits, labels

def compute_metrics(pred):
    logits, labels, loss = pred
    perplexity = np.exp(loss)
    return {
        'loss': loss.mean(),
        'perplexity': perplexity.mean()
    }

class TrainingCallback(TrainerCallback):
    def __init__(self):
        self.step_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        global BATCH_SIZE
        global FULL_BATCH_SIZE
        self.step_bar = tqdm(desc=f'Epoch {int(state.epoch)+1}', total=ceil(FULL_BATCH_SIZE//BATCH_SIZE//args.gradient_accumulation_steps), file=sys.stdout, leave=True)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_bar.update(1)

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        self.step_bar.close()

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\nMax memory allocated: {torch.cuda.max_memory_allocated()}")

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        pass

class CustomTrainer(Trainer):
    def create_progress_bar(self, *args, **kwargs):
        return tqdm(*args, **kwargs, file=sys.stdout)

config = LinformerConfig(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            seq_len=SEQ_LENGTH,
            depth=8,
            k=128,
            heads=8,
            dim_head=None,
            one_kv_head=False,
            share_kv=False,
            reversible=False,
            dropout=0.1
)

model = LinformerLM(config)

"""config = TransformerConfig(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            depth=8,
            heads=8,
            dropout=0.1
)

model = TransformerLM(config)"""

training_args = TrainingArguments(
    output_dir=f'./pretrained_model/{SEQ_LENGTH}',
    logging_dir=f'./logs/{SEQ_LENGTH}',
    report_to='tensorboard',
    log_level='error',
    disable_tqdm=True,
    num_train_epochs=50,
    weight_decay=0.01,
    fp16=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_accumulation_steps=200,
    overwrite_output_dir=False,
    eval_strategy='steps',
    save_strategy='epoch',
    eval_steps=25,
    logging_strategy='steps',
    logging_steps=10,
    include_for_metrics=['loss'],
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    callbacks=[TrainingCallback()],
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.remove_callback(PrinterCallback)

trainer.train()

model.save_pretrained('./')
tokenizer.save_pretrained('./')

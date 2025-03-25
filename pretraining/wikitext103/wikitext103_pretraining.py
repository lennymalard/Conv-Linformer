import os
import torch.cuda
from data.utils import MLMDataset
from transformers import RobertaTokenizerFast, TrainingArguments, TrainerCallback, Trainer, default_data_collator, logging
from linformer.linformer import *
from transformer.transformer import *
from conv_linformer.conv_linformer import *
from tqdm import tqdm
import sys, os
import numpy as np
from transformers.trainer_callback import PrinterCallback, EarlyStoppingCallback
from math import ceil

BATCH_SIZE = None
FULL_BATCH_SIZE = None
SEQ_LENGTHS = [256, 512]
K_DIMS = [32, 64, 128, 256]
LEARNING_RATES = [5e-5, 1e-5]

def arch_selector(arch, configs):
    if arch == 'transformer' and arch in configs.keys():
        return TransformerLM(configs[arch])
    elif arch == 'linformer' and arch in configs.keys():
        return LinformerLM(configs[arch])
    elif arch == 'conv_linformer' and arch in configs.keys():
        return ConvLinformerLM(configs[arch])
    else:
        raise ValueError('Unknown architecture {}'.format(arch))

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

    def on_epoch_begin(self, args, state, control, **kwargs):
        global BATCH_SIZE
        global FULL_BATCH_SIZE
        self.step_bar = tqdm(desc=f'Epoch {int(state.epoch)+1}', total=ceil(FULL_BATCH_SIZE//BATCH_SIZE//args.gradient_accumulation_steps), file=sys.stdout, leave=True)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_bar.update(1)

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        self.step_bar.close()

    def on_train_end(self, args, state, control, **kwargs):
        memory_allocated = f"\nMax memory allocated: {torch.cuda.max_memory_allocated()}"
        print(memory_allocated)
        with open(args.logging_dir + '/max_memory.txt', 'w') as f:
            f.write(memory_allocated)

class CustomTrainer(Trainer):
    def create_progress_bar(self, *args, **kwargs):
        return tqdm(*args, **kwargs, file=sys.stdout)

def training_pipeline(architecture, seq_length, batch_size, learning_rate=1e-5, k_dim=None):
    torch.cuda.empty_cache()

    print(f"Starting training...\nArchitecture: {architecture}, Seq Length: {seq_length}, Batch Size: {batch_size}, k dim: {k_dim}, Learning Rate: {learning_rate}")

    if os.path.exists(f'./{architecture}/{learning_rate}/{seq_length}/pretrained_model/logs/max_memory.txt') if architecture == 'transformer' \
            else os.path.exists(f'./{architecture}/{learning_rate}/{seq_length}/k_{k_dim}/pretrained_model/logs/max_memory.txt'):
        print("Model already trained.")
        return

    global BATCH_SIZE
    global FULL_BATCH_SIZE

    assert architecture in ['linformer', 'transformer', 'conv_linformer']
    assert seq_length in [128, 256, 512, 1024]

    if architecture == 'transformer' and k_dim is not None:
        raise ValueError('k_dim cannot be used with transformer architecture')
    elif (architecture == 'linformer' or architecture == 'conv_linformer') and k_dim is None:
        raise ValueError('k_dim has to be used with linformer architecture')

    training_dataset = MLMDataset(F'../../data/pretraining/wikitext50/wikitext103_{seq_length}.hdf5')
    validation_dataset = MLMDataset(F'../../data/pretraining/wikitext2/wikitext2_{seq_length}_validation.hdf5')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    BATCH_SIZE = batch_size
    FULL_BATCH_SIZE = len(training_dataset)

    configs = {
        'transformer': TransformerConfig(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            depth=8,
            heads=8,
            dropout=0.1
        ),
        'linformer': LinformerConfig(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            seq_len=seq_length,
            depth=8,
            k=k_dim,
            heads=8,
            dim_head=None,
            one_kv_head=False,
            share_kv=False,
            reversible=False,
            dropout=0.1
        ),
        'conv_linformer': ConvLinformerConfig(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            seq_len=seq_length,
            depth=8,
            k=k_dim,
            heads=8,
            dim_head=None,
            one_kv_head=False,
            share_kv=False,
            reversible=False,
            dropout=0.1
        )

    }

    model = arch_selector(architecture, configs)

    training_args = TrainingArguments(
        output_dir=f'./{architecture}/{learning_rate}/{seq_length}/pretrained_model/outputs' if architecture == 'transformer'
        else f'./{architecture}/{learning_rate}/{seq_length}/k_{k_dim}/pretrained_model/outputs',

        logging_dir=f'./{architecture}/{learning_rate}/{seq_length}/pretrained_model/logs' if architecture == 'transformer'
        else f'./{architecture}/{learning_rate}/{seq_length}/k_{k_dim}/pretrained_model/logs',

        report_to='tensorboard',
        log_level='error',
        disable_tqdm=True,
        num_train_epochs=20,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        fp16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=200,
        overwrite_output_dir=False,
        eval_strategy='steps',
        save_strategy='epoch',
        save_steps=2,
        eval_steps=100,
        logging_strategy='steps',
        logging_steps=100,
        include_for_metrics=['loss'],
        metric_for_best_model='perplexity',
        greater_is_better=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[TrainingCallback(), EarlyStoppingCallback(early_stopping_patience=500)],
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.remove_callback(PrinterCallback)

    if not os.listdir(f'./{architecture}/{learning_rate}/{seq_length}/pretrained_model/outputs' if architecture == 'transformer'
        else f'./{architecture}/{learning_rate}/{seq_length}/k_{k_dim}/pretrained_model/outputs'):
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)

    if architecture == 'transformer':
        model.save_pretrained(f'./{architecture}/{learning_rate}/{seq_length}/pretrained_model')
        tokenizer.save_pretrained(f'./{architecture}/{learning_rate}/{seq_length}/pretrained_model')
    elif architecture == 'linformer' or 'conv_linformer':
        model.save_pretrained(f'./{architecture}/{learning_rate}/{seq_length}/k_{k_dim}/pretrained_model')
        tokenizer.save_pretrained(f'./{architecture}/{learning_rate}/{seq_length}/k_{k_dim}/pretrained_model')

if __name__ == '__main__':
    for lr in LEARNING_RATES:
        batch_size = 8
        for seq_length in SEQ_LENGTHS:
            for k in K_DIMS:
                if (seq_length == 256 and k == 256) or (seq_length == 512 and k == 32):
                    continue
                training_pipeline('linformer', seq_length=seq_length, batch_size=batch_size, k_dim=k, learning_rate=lr)
                training_pipeline('conv_linformer', seq_length=seq_length, batch_size=batch_size, k_dim=k, learning_rate=lr)
            training_pipeline('transformer', seq_length=seq_length, batch_size=batch_size, learning_rate=lr)
            batch_size/=2



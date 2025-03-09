import numpy as np

from data.utils import MLMDataset
from transformers import RobertaTokenizerFast, TrainingArguments, TrainerCallback, Trainer, default_data_collator
from torch.utils.data import DataLoader
from linformer.linformer import *

SEQ_LENGTH = 128

training_dataset = MLMDataset(F'../../data/pretraining/wikitext2/wikitext2_{SEQ_LENGTH}.hdf5')
validation_dataset = MLMDataset(F'../../data/pretraining/wikitext2/wikitext2_{SEQ_LENGTH}_validation.hdf5')
dataloader = DataLoader(training_dataset, batch_size=4)
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def compute_metrics(pred):
    logits, labels, loss = pred
    perplexity = np.exp(loss)
    return {
        'loss': loss,
        'perplexity': perplexity
    }

class MemoryCallback(TrainerCallback):
    def __init__(self):
        self.max_memory = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 ** 2
            self.max_memory = max(current_memory, self.max_memory)
            print(f"Epoch: {state.epoch} - Current memory allocated: {self.max_memory:.2f}MB")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Max memory allocated: {self.max_memory}")

config = LinformerConfig(
            num_tokens=tokenizer.vocab_size,
            dim=512,
            seq_len=128,
            depth=4,
            k=128,
            heads=8,
            dim_head=None,
            one_kv_head=False,
            share_kv=False,
            reversible=False,
            dropout=0.1
)

model = LinformerLM(config)

training_args = TrainingArguments(
    output_dir=f'./pretrained_model/{SEQ_LENGTH}',
    logging_dir=f'./logs/{SEQ_LENGTH}',
    report_to='tensorboard',
    num_train_epochs=50,
    weight_decay=0.01,
    fp16=True,
    overwrite_output_dir=False,
    eval_strategy='steps',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=25,
    include_for_metrics=['loss'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    callbacks=[MemoryCallback()],
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
)

trainer.train()

model.save_pretrained('./')
tokenizer.save_pretrained('./')
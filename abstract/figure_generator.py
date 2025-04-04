import matplotlib.pyplot as plt
import matplotlib
import os
from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#matplotlib.use('TkAgg')

PATH = "/home/lenny/Documents/DEV/git/lilformer/pretraining/wikitext103"

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-Green
    '#17becf'   # Cyan
])

def find_log_dir(path):
    log_dirs = []
    for root, dirs, files in os.walk(path):
        if os.path.basename(os.path.dirname(root)) == '.git':
            continue
        if os.path.basename(root) == 'logs':
            log_dirs.append(root)
    return log_dirs

log_dirs = find_log_dir(PATH)

def k_formatter(x, pos):
    if x >= 1000:
        return f'{x*1e-3:.0f}k'
    return x

def plot(directories, tag, architectures, learning_rates, sequence_lengths, title='', label=''):
    for dir in log_dirs:
        dir_split = dir.split(os.sep)[:-2]

        while dir_split[0] != "wikitext103":
            dir_split.pop(0)
        dir_split.pop(0)

        if dir_split[0] == 'transformer':
            arch, lr, seq_len = dir_split
            k_dim = None
        else:
            arch, lr, seq_len, k_dim = dir_split
            k_dim = k_dim[2:]

        if arch not in architectures or lr not in learning_rates or seq_len not in sequence_lengths:
            continue

        event_acc = EventAccumulator(dir)
        event_acc.Reload()
        try:
            loss_events = event_acc.Scalars(tag)
        except KeyError:
            continue
        steps = [event.step for event in loss_events]
        losses = [event.value for event in loss_events]

        plt.plot(steps, losses, label=f'{arch}, k_dim: {k_dim}', linestyle='-' if arch == 'transformer' else '--' if arch == 'linformer' else '-.')

    plt.ylim(0, 1500)
    plt.gca().xaxis.set_major_formatter(k_formatter)
    plt.text(-10, 50, label, fontsize=12)
    plt.xlabel('step')
    plt.ylabel(tag.split(os.sep)[1])
    plt.title(title)
    plt.legend(fontsize=9)
    plt.show()

plot(directories=log_dirs,
     tag='eval/perplexity',
     architectures=['linformer', 'transformer', 'conv_linformer'],
     learning_rates=['5e-05'],
     sequence_lengths=['512'],
     title='Eval perplexity, seq_len: 512, lr: 5e-05',
     label='(a)'
     )

plot(directories=log_dirs,
     tag='eval/perplexity',
     architectures=['linformer', 'transformer', 'conv_linformer'],
     learning_rates=['1e-05'],
     sequence_lengths=['512'],
     title='Eval perplexity, seq_len: 512, lr: 1e-05',
     label='(b)'
     )
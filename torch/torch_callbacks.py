import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# https://github.com/leigh-plt/cs231n_hw2018/blob/master/assignment2/pytorch_tutorial.ipynb
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

class torch_callback:
    def __init__(self, monitor, check_method, early_stopping_patience, verbose=True):
        assert(check_method in ['min', 'max']), 'only support min max'
        self.best_history = {}
        self.epoch_history = defaultdict(list)
        self.batch_mean = {}
        self.batch_history = defaultdict(self.get_default_batch_history)
        self.epoch = 1
        self.monitor = monitor
        self.best_monitor_point = np.inf*(1 if check_method == 'min' else -1)
        self.check_method = eval(check_method)
        self.not_improve_epochs = 0
        self.continue_not_improve_epochs = 0
        self.reach_best = False
        self.reach_early_stopping = False
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
    def reset_epoch_history(self):
        self.epoch_history = defaultdict(list)
    def reset_batch_history(self):
        self.batch_history = defaultdict(self.get_default_batch_history)
    def update_batch_history(self, metric_name, metric_value, batch_size):
        self.batch_history[metric_name]['count'] += batch_size
        self.batch_history[metric_name]['value'] += batch_size*metric_value
    def get_default_batch_history(self):
        return {'count':0, 'value':0}
    def compute_batch_mean(self):
        self.batch_mean = {}
        for key, value in self.batch_history.items():
            self.batch_mean[key] = value['value'] / value['count']
    def on_epoch_end(self):
        self.compute_batch_mean()
        self.epoch_history['epoch'].append(self.epoch)
        report = f'Epoch {self.epoch}'
        for key, value in self.batch_mean.items():
            self.epoch_history[key].append(value)
            report += f' -- {key}:{value:.4f}'
        if self.verbose:
            print(report)
        self.epoch += 1
        self.best_monitor_point = self.check_method(self.epoch_history[self.monitor])
        if self.epoch_history[self.monitor][-1] == self.best_monitor_point:
            # current is best
            self.reach_best = True
            self.continue_not_improve_epochs = 0
        else:
            # not improve
            self.reach_best = False
            self.not_improve_epochs += 1
            self.continue_not_improve_epochs += 1
            if self.early_stopping_patience == self.not_improve_epochs:
                self.reach_early_stopping = True
    def on_epoch_start(self):
        self.reset_batch_history()
    def get_history_df(self):
        return pd.DataFrame(self.epoch_history)
    def plot_history(self, prefix):
        history_df = self.get_history_df()
        cols = history_df.columns.tolist()
        val_cols = [item for item in filter(lambda x: x.startswith('val'), cols)]
        tra_cols = [item.replace('val_', '') for item in val_cols]
        def plot_train_valid(df, train_col, valid_col, prefix=prefix):
            plt.clf()
            plt.plot(df['epoch'], df[train_col])
            plt.plot(df['epoch'], df[valid_col])
            plt.grid()
            plt.legend()
            plt.show()
            plt.savefig(f'{prefix}_{train_col}.png')
        for train_col, valid_col in zip(tra_cols, val_cols):
            plot_train_valid(history_df, train_col, valid_col)
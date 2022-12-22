import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def topk_correct(output, target, k=1):
    _, pred = output.topk(k, dim=1)
    tiled = torch.tile(target, (k, 1))
    return torch.sum(tiled.eq(pred.t())).item()


class Log(object):
    def __init__(self, log_dir, log_name, fields):
        self.fields = fields
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(self.log_dir, f'{self.log_name}.csv')

        with open(self.log_path, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=self.fields)
            csv_writer.writeheader()

    def append(self, **kwargs):
        row = {field: None for field in self.fields}

        for key, val in kwargs.items():
            if key in self.fields:
                row[key] = val
            else:
                raise ValueError(f'\'{key}\' is not a valid log field')

        with open(self.log_path, 'a+') as f:
            csv_writer = csv.DictWriter(f, fieldnames=self.fields)
            csv_writer.writerow(row)


class Confusion(object):
    def __init__(self, device, classes=10):
        self.classes = classes
        self.matrix = torch.zeros((classes, classes),
                                  dtype=torch.int64,
                                  device=device)

    def add(self, pred, target):
        assert torch.numel(pred) == torch.numel(target)

        for i in range(torch.numel(pred)):
            self.matrix[target[i].item(), pred[i].item()] += 1

    def accuracy(self):
        return torch.sum(torch.diag(self.matrix)).item() \
            / torch.sum(self.matrix).item()

    def correct(self):
        return torch.sum(torch.diag(self.matrix)).item()

    def tpr(self):
        return torch.div(torch.diag(self.matrix),
                         torch.sum(self.matrix, 1))

    def precision(self):
        return torch.div(torch.diag(self.matrix),
                         torch.sum(self.matrix, 0))

    def get(self, normalize=False):
        if normalize:
            return torch.divide(self.matrix, torch.sum(self.matrix)).cpu().numpy()
        else:
            return self.matrix.cpu().numpy()

    def plot(self, title, labels, heatmap_kwargs=None, subplot_kwargs=None):
        heatmap_kwargs = heatmap_kwargs or {}
        subplot_kwargs = subplot_kwargs or {}
        fig, ax = plt.subplots(**subplot_kwargs)
        c = self.get()
        sns.heatmap(c, **heatmap_kwargs)
        ax.set_title(r'{} Accuracy$={}/{}={:.2f}\%$'.format(title,
                                                            self.correct(),
                                                            torch.sum(self.matrix).item(),
                                                            self.accuracy() * 100))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_ylim((0, self.classes))
        ax.set_xlim((0, self.classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        fig.tight_layout()
        return fig, ax

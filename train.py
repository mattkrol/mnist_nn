#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Net
from util import Confusion, Log, topk_correct


def train(args, model, device, data_loader, optimizer, epoch):
    model.train()
    batch_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch {} [{}/{} ({:.0f}%)], Loss: {:.9f}'.format(epoch,
                                                                          batch_idx * len(data),
                                                                          len(data_loader.dataset),
                                                                          100. * batch_idx / len(data_loader),
                                                                          loss.item()))
        if args.dry_run:
            break
    return batch_loss / len(data_loader)


def test(model, device, data_loader, title):
    model.eval()
    test_loss = 0
    correct = 0
    confusion = Confusion(device, classes=10)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += topk_correct(output, target, k=1)
            pred = output.argmax(dim=1, keepdim=True)
            confusion.add(pred.flatten(), target.flatten())
    test_loss /= len(data_loader.dataset)
    test_acc = correct / len(data_loader.dataset)
    print('{}, Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(title, test_loss, correct,
                                                                         len(data_loader.dataset),
                                                                         100. * test_acc))
    return test_loss, test_acc, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    results_dir = os.path.join(os.getcwd(), 'results')
    os.mkdir(results_dir)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, **train_kwargs)
    test_loader = DataLoader(test_data, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fields = ['epoch', 'batch_train_loss', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
    log = Log(results_dir, 'train', fields)

    batch_train_loss_arr = []
    train_loss_arr = []
    train_acc_arr = []
    test_loss_arr = []
    test_acc_arr = []

    train_loss, train_acc, train_c = test(model, device, train_loader, 'Train Set')
    train_loss_arr.append(train_loss)
    train_acc_arr.append(train_acc)

    test_loss, test_acc, test_c = test(model, device, test_loader, 'Test Set')
    test_loss_arr.append(test_loss)
    test_acc_arr.append(test_acc)
    for epoch in range(1, args.epochs + 1):
        batch_train_loss = train(args, model, device, train_loader, optimizer, epoch)
        batch_train_loss_arr.append(batch_train_loss)

        train_loss, train_acc, train_c = test(model, device, train_loader, 'Train Set')
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)

        test_loss, test_acc, test_c = test(model, device, test_loader, 'Test Set')
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)

        d = {
            'epoch': epoch,
            'batch_train_loss': batch_train_loss,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        log.append(**d)

    fig, ax1 = plt.subplots()

    l0 = ax1.plot(np.arange(1, args.epochs + 1), batch_train_loss_arr, 'c', label='Batch Train Loss')
    l1 = ax1.plot(np.arange(0, args.epochs + 1), train_loss_arr, 'm', label='Train Loss')
    l2 = ax1.plot(np.arange(0, args.epochs + 1), test_loss_arr, 'g', label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_xticks(np.arange(0, args.epochs + 1))
    ax1.grid(axis='x')

    ax2 = ax1.twinx()
    l3 = ax2.plot(np.arange(0, args.epochs + 1), train_acc_arr, 'b', label='Train Accuracy')
    l4 = ax2.plot(np.arange(0, args.epochs + 1), test_acc_arr, 'r', label='Test Accuracy')
    ax2.grid()
    ax2.set_ylabel('Accuracy')
    ax2.set_yticks(np.arange(0, 1.05, 0.05))
    lines = l0 + l1 + l2 + l3 + l4
    labels = [ln.get_label() for ln in lines]
    ax2.legend(lines, labels, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode='expand', ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_vs_acc.png'))

    labels = [i for i in range(10)]

    heatmap_kwargs = {'annot': True, 'square': False}
    subplot_kwargs = {'figsize': (12, 10)}

    train_c.plot('Train Set', labels,
                 heatmap_kwargs=heatmap_kwargs,
                 subplot_kwargs=subplot_kwargs)
    plt.savefig(os.path.join(results_dir, 'train_conf_mat.png'))

    test_c.plot('Test Set', labels,
                heatmap_kwargs=heatmap_kwargs,
                subplot_kwargs=subplot_kwargs)
    plt.savefig(os.path.join(results_dir, 'test_conf_mat.png'))

    torch.save(model.state_dict(), os.path.join(results_dir, 'net.pt'))


if __name__ == '__main__':
    main()

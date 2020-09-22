import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import TrainDataset, ValidationDataset, DataLoader, get_cifar100_dataset
import torch.nn as nn
import torch.optim as optim
from model import VGGModel
import os
from basisModel import basisModel, display_stats

def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, -1)

    return torch.mean((y_argmax==y).type(torch.float))


def train(model, data_loader, criterion, optimizer):
    model.train()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []
    for i, sample in enumerate(data_loader):
        x, y = sample

        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = get_accuracy(y_pred, y)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 100 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))), flush=True)

    print('Finished training. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))), flush=True)

    return float(np.mean(losses)), float(np.mean(accs))


def validation(model, data_loader, criterion):
    model.eval()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []
    for i, sample in enumerate(data_loader):
        x, y = sample

        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = get_accuracy(y_pred, y)

        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 100 == 0:
            print('Finished validating %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))), flush=True)

    print('Finished validation. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))), flush=True)

    return float(np.mean(losses)), float(np.mean(accs))


def run_experiment():
    model = VGGModel(n_classes=config.n_classes)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    # load in old weights
    load_model_id = 2
    load_file_path = os.path.join('./SavedModels/Run%d/' % load_model_id, 'model.pth')
    model.load_state_dict(torch.load(load_file_path)['state_dict'])

    # creates the basis model
    model = basisModel(model, True, True, True)#basisModel(model, opts.use_weights, opts.add_bn, opts.fixed_basbs)
    model.update_channels(0.8)
    
    display_stats(model, (64,64))

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # creates the datasets
    tr_dataset = get_cifar100_dataset('./data/', True, download=True)  # TrainDataset()  # A custom dataloader may be needed, in which case use TrainDataset()
    val_dataset = get_cifar100_dataset('./data/', False, download=True)  # ValidationDataset() # A custom dataloader may be needed, in which case use ValidationDataset()
   
    tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_acc = 0
    for epoch in range(1, config.n_epochs + 1):
        print('Epoch', epoch)

        losses, acc = train(model, tr_dataloader, criterion, optimizer)

        losses, acc = validation(model, val_dataloader, criterion)
        #scheduler.step()

        if acc > best_acc:
            print('Model Improved -- Saving.')
            best_loss = losses

            #save_file_path = os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(epoch, losses))
            save_file_path = os.path.join(config.save_dir, 'model.pth')
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            try:
                os.mkdir(config.save_dir)
            except:
                pass

            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path), flush=True)

        print('Training Finished') 


if __name__ == '__main__':
    run_experiment()

import config
import torch
import numpy as np
from dataloader import TrainDataset, ValidationDataset, DataLoader, get_cifar100_dataset
from model import VGGModel
import os
import time

if config.use_tensorRT:
    from torch2trt import torch2trt


def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, -1)

    return torch.mean((y_argmax==y).type(torch.float))
    

def validation(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

    if config.use_tensorRT:
        # Get random input to pass as a sample to TensorRT
        x, _ = next(iter(data_loader))

        if config.use_cuda:
            x = x.cuda()
        else:
            raise RuntimeError('Cannot use TensorRT without CUDA')

        # Optimize
        model = torch2trt(model, [x])
    
    total_imgs = 0
    times, accs = [], []

    for i, sample in enumerate(data_loader):
        x, y = sample
        total_imgs += x.shape[0]
        
        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        with torch.no_grad():
            start_time = time.time()
            y_pred = model(x)
            end_time = time.time()
            times.append((end_time-start_time)/float(x.shape[0]) * 1000 * 1000)  # saves the average time per image

        acc = get_accuracy(y_pred, y)  # computes the accuracy per batch

        accs.append(acc.item())

    print('Finished evaluating over % images \n'
          'Avg Time per Image: %.4f(micro-sec). Accuracy: %.4f' % (total_imgs, float(np.mean(times)), float(np.mean(accs)*100)), flush=True)

    return float(np.mean(times)), float(np.mean(accs))
    

def evaluate():
    model = VGGModel(n_classes=config.n_classes)
    
    val_dataset = get_cifar100_dataset('./data/', False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    save_file_path = os.path.join(config.save_dir, 'model.pth')
    model.load_state_dict(torch.load(save_file_path)['state_dict'])

    if config.use_tensorRT:
        print('Evaluating model using TensorRT')
    else:
        print('Evaluating model without using TensorRT')

    avg_time, avg_acc = validation(model, val_dataloader)
    
    
if __name__ == '__main__':
    evaluate()

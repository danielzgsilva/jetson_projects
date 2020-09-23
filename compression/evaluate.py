import os


#os.environ['CUDA_LAUNCH_BLOCKING']='1'

import config
import torch
import numpy as np
from dataloader import TrainDataset, ValidationDataset, DataLoader, get_cifar100_dataset
from model import VGGModel, VGGModel_old
import time

from basisModel import basisModel, display_stats
from options import Options

opts = Options().parse()

if opts.tensorRT:
    from torch2trt import torch2trt


def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, -1)

    return torch.mean((y_argmax==y).type(torch.float))
    

def validation(model, data_loader, opts):
    model.eval()

    if opts.compress:
        print('Compressing model with basis filter algorithm, compression factor of {}'.format(opts.compress_factor))
        model = basisModel(model, opts.use_weights, opts.add_bn, opts.fixed_basbs)
        model.update_channels(opts.compress_factor)
        display_stats(model, (64,64))

    else:
        print('No compression schema')
    
    if config.use_cuda:
        model.cuda()
    
    if opts.tensorRT:
        print('Optimizing model with TensorRT')

        # Get random input to pass as a sample to TensorRT
        x, _ = next(iter(data_loader))

        if config.use_cuda:
            x = x.cuda()
        else:
            raise RuntimeError('Cannot use TensorRT without CUDA')

        # Optimize
        trt_model = torch2trt(model, [x], max_batch_size=config.batch_size)

        del model
        del x
        torch.cuda.empty_cache()
        model = trt_model
        model.cuda()

    else:
        print('No TensorRT')
    
    print('memory usage:')
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_summary())
 
    print('Evaluating model with {} iterations over {} images'.format(opts.n, len(data_loader)*config.batch_size))
    all_times, all_accs = [], []

    for i in range(opts.n):
        times, accs = [], []

        for _, sample in enumerate(data_loader):
            x, y = sample
        
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

        iteration_time, iteration_acc = float(np.mean(times)), float(np.mean(accs))*100
        all_times.append(iteration_time)
        all_accs.append(iteration_acc)
        print('Iteration %d: Avg Time per Image: %.4f (micro-sec) Accuracy: %.4f' % (i, iteration_time, iteration_acc), flush=True)
    
    avg_time, avg_acc = float(np.mean(all_times[1:])), float(np.mean(all_accs))
    
    print('-'*70)
    print('Final reuslts: Avg Time per Image: %.4f (micro-sec) Accuracy: %.4f' % (avg_time, avg_acc), flush=True)
    return avg_time, avg_acc
    

def evaluate(opts):
    val_dataset = get_cifar100_dataset('./data/', False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
    
    save_file_path = os.path.join(opts.save_dir, opts.model)

    if opts.load_state_dict:
        if opts.use_vgg_old:
            model = VGGModel_old(n_classes=config.n_classes)
        else:
            model = VGGModel(n_classes=config.n_classes)

        model.load_state_dict(torch.load(save_file_path)['state_dict'])

    else:
        model = torch.load(save_file_path)
    
    avg_time, avg_acc = validation(model, val_dataloader, opts)
     
    
if __name__ == '__main__':
    evaluate(opts)



#os.environ['CUDA_LAUNCH_BLOCKING']='1'

from engine import train_one_epoch, evaluate
import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import FlirDataset
from options import Options

import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import utils

opts = Options().parse()

if opts.tensorRT:
    from torch2trt import torch2trt


# def validation(model, data_loader, opts):
#     model.eval()
#
#     if opts.tensorRT:
#         print('Optimizing model with TensorRT')
#
#         # Get random input to pass as a sample to TensorRT
#         x, _ = next(iter(data_loader))
#
#         if config.use_cuda:
#             x = x.cuda()
#         else:
#             raise RuntimeError('Cannot use TensorRT without CUDA')
#
#         # Optimize
#         trt_model = torch2trt(model, [x], max_batch_size=config.batch_size)
#
#         del model
#         del x
#         torch.cuda.empty_cache()
#         model = trt_model
#         model.cuda()
#
#     else:
#         print('No TensorRT')
#
#     print('memory usage:')
#     print(torch.cuda.memory_allocated())
#     print(torch.cuda.memory_summary())
#
#     print('Evaluating model with {} iterations over {} images'.format(opts.n, len(data_loader)*config.batch_size))
#
#     evaluate(model, data_loader_test, device=device)

    # all_times, all_accs = [], []
    #
    # for i in range(opts.n):
    #     times, accs = [], []
    #
    #     for _, sample in enumerate(data_loader):
    #         x, y = sample
    #
    #         if config.use_cuda:
    #             x = x.cuda()
    #             y = y.cuda()
    #
    #         with torch.no_grad():
    #             start_time = time.time()
    #             y_pred = model(x)
    #             end_time = time.time()
    #             times.append((end_time-start_time)/float(x.shape[0]) * 1000 * 1000)  # saves the average time per image
    #
    #         acc = get_accuracy(y_pred, y)  # computes the accuracy per batch
    #
    #         accs.append(acc.item())
    #
    #     iteration_time, iteration_acc = float(np.mean(times)), float(np.mean(accs))*100
    #     all_times.append(iteration_time)
    #     all_accs.append(iteration_acc)
    #     print('Iteration %d: Avg Time per Image: %.4f (micro-sec) Accuracy: %.4f' % (i, iteration_time, iteration_acc), flush=True)
    #
    # avg_time, avg_acc = float(np.mean(all_times[1:])), float(np.mean(all_accs))
    #
    # print('-'*70)
    # print('Final reuslts: Avg Time per Image: %.4f (micro-sec) Accuracy: %.4f' % (avg_time, avg_acc), flush=True)
    # return avg_time, avg_acc
    

def validation(opts):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    val_dataset = FlirDataset(data_root=opts.data_root, validation=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4,
                                                   collate_fn=utils.collate_fn_tr)

    file_path = os.path.join(opts.save_dir, opts.model)
    model = torch.load(file_path)
    model.to(device)

    model.eval()

    if opts.tensorRT:
        print('Optimizing model with TensorRT')

        # Get random input to pass as a sample to TensorRT
        x, _ = next(iter(val_loader))

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

    print('Evaluating model with {} iterations over {} images'.format(opts.n, len(val_loader) * config.batch_size))

    evaluate(model, val_loader, device=device)
     
    
if __name__ == '__main__':
    validation(opts)



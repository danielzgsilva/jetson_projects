from engine import train_one_epoch, evaluate
import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import FlirDataset

import torch.nn as nn
import torch.optim as optim
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import utils


def run_experiment():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tr_dataset = FlirDataset()  
    val_dataset = FlirDataset(validation=True)
    
    data_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=utils.collate_fn_tr)
    data_loader_test = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4, collate_fn=utils.collate_fn_tr)

    model = fasterrcnn_resnet50_fpn(num_classes=6) 
    
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
        try:
            os.mkdir(config.save_dir)
        except:
            pass
        
        save_file_path = os.path.join(config.save_dir, 'model.pth')
        torch.save(model, save_file_path)
        print('Model saved ', str(save_file_path), flush=True)


if __name__ == '__main__':
    run_experiment()

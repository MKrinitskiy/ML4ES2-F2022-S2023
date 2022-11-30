import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple, List, Type, Dict, Any

class Perceptron(torch.nn.Module):
    
    def __init__(self, 
                 input_resolution: Tuple[int, int] = (28, 28),
                 input_channels: int = 1, 
                 hidden_layer_features: List[int] = [256, 256],
                 activation: Type[torch.nn.Module] = torch.nn.ReLU,
                 num_classes: int = 10):

        super().__init__()

        network_layers = []
        in_features = np.product(input_resolution) * input_channels

        for out_features in hidden_layer_features:
            network_layers.append(torch.nn.Linear(in_features, out_features))
            network_layers.append(activation())
            in_features = out_features
            
        network_layers.append(torch.nn.Linear(in_features, num_classes))
        self._layers = torch.nn.Sequential(*network_layers)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self._layers.forward(x)


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, 
                       loss_function: torch.nn.Module, 
                       data_loader: torch.utils.data.DataLoader,
                       tb_writer: SummaryWriter,
                       epoch: int,
                       batch_size: int):
    train_loss = []
    batch_averaged_loss = []
    model.train()
    for idx,(batch_data, batch_labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()
        data_gpu, labels_gpu = batch_data.cuda(), batch_labels.cuda()
        output = model(data_gpu)
        loss = loss_function(output, labels_gpu)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        batch_averaged_loss.append(loss.item())
        if idx % 20 == 0:
            batch_averaged_loss_ = np.sum(batch_averaged_loss)/len(batch_averaged_loss)
            tb_writer.add_scalar('train_batch_loss', batch_averaged_loss_, global_step = epoch*(int(len(data_loader.dataset)/batch_size))+idx)
    
    for tag, param in model.named_parameters():
        tb_writer.add_histogram('grad/%s'%tag, param.grad.data.cpu().numpy(), epoch)
        tb_writer.add_histogram('weight/%s' % tag, param.data.cpu().numpy(), epoch)
    
    return train_loss


def validate_single_epoch(model: torch.nn.Module,
                          loss_function: torch.nn.Module, 
                          data_loader: torch.utils.data.DataLoader):
    
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, labels in tqdm(data_loader, total=len(data_loader)):
            data_gpu, labels_gpu = data.cuda(), labels.cuda()
            output = model(data_gpu)
            test_loss += loss_function(output, labels_gpu).sum()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels_gpu.view_as(pred)).sum()

    return {'loss': test_loss.item() / len(data_loader.dataset),
            'accuracy': correct.cpu().numpy() / len(data_loader.dataset)}
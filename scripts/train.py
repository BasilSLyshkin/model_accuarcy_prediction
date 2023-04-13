import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from scripts.model import LanguageModel
import numpy as np

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses,train_accs, val_losses,val_accs):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
    Calculate train and validation perplexities given lists of losses
    """
    train_perplexities, val_perplexities = np.exp(train_losses), np.exp(val_losses)

    axs[1].plot(range(1, len(train_accs) + 1), train_accs, label='train')
    axs[1].plot(range(1, len(val_accs) + 1), val_accs, label='val')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, metric, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for _, indices, lengths, targets in tqdm(loader, desc=tqdm_desc):
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Process one training step: calculate loss,
        call backward and make one optimizer step.
        Accumulate sum of losses for different batches in train_loss
        """

        optimizer.zero_grad()
        indices = indices[:,:lengths.max()].to(device).long()
        targets = targets.to(device)
        logits = model(indices, lengths)
        loss = criterion(logits, targets)
        acc = metric(logits, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * indices.shape[0]
        train_acc += acc.item() * indices.shape[0]

    train_loss /= len(loader.dataset)
    train_acc /= len(loader.dataset)
    return train_loss, train_acc


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, metric,tqdm_desc: str):
    
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    
    device = next(model.parameters()).device
    val_loss = 0.0
    val_acc = 0.0
    
    model.eval()
    for _, indices, lengths, targets in tqdm(loader, desc=tqdm_desc):

        indices = indices[:,:lengths.max()].to(device).long()
        logits = model(indices, lengths)
        targets = targets.to(device)
        loss = criterion(logits, targets)
        acc = metric(logits, targets)
        
        val_loss += loss.item() * indices.shape[0]
        val_acc += acc * indices.shape[0]

    val_loss /= len(loader.dataset)
    val_acc /= len(loader.dataset)

    return val_loss, val_acc


def train(model: LanguageModel,model_name: str, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    criterion = nn.CrossEntropyLoss()
    metric = lambda logits, targets: (logits.argmax(1) == targets).float().mean().cpu()
    previous_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = training_epoch( 
            model, optimizer, criterion, train_loader, metric,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss, val_acc = validation_epoch(
            model, criterion, val_loader,metric,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()
            
        if previous_val_loss>val_loss:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
            torch.save(checkpoint, f'models/{model_name}.pth')
            previous_val_loss = val_loss
            
        train_losses += [train_loss]
        train_accs += [train_acc]
        val_losses += [val_loss]
        val_accs += [val_acc]

        plot_losses(train_losses,train_accs, val_losses,val_accs)

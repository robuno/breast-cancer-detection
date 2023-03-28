import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import os
import zipfile
from pathlib import Path
import requests
import os

def single_bar_plot_dataset_53(labels, values, title, xlabel, ylabel, bar_color="orange"):
    fig = plt.figure(figsize = (5, 3))
 
    plt.bar(labels, values, 
            width = 0.4, color = bar_color)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)



def plot_loss_accuracy_grid2(results):
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss", color = "red")
    plt.plot(epochs, test_loss, label="test_loss", color = "blue")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy", color = "red")
    plt.plot(epochs, test_accuracy, label="test_accuracy", color = "blue")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
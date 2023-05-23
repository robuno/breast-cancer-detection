import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2 as cv
import seaborn as sns

import torch
from torch import nn

import os
import zipfile
from pathlib import Path
import requests

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


def plot_mamm_images(patient_id, df):
    # get a patient's image with patient ID

    # selected_patient_id = 53727
    selected_patient_id = patient_id
    selected_patient_df = df.loc[df["patient_id"] == selected_patient_id]
    # print(selected_patient_df)
    image_count_selected_patient = len(df.loc[df["patient_id"] == selected_patient_id])

    img_links = []
    for index, row in selected_patient_df.iterrows():
        img_file_name = str(row["patient_id"])+"_"+str(row["image_id"])+".png"
        # print('/images/'+img_file_name)
        img = np.asarray(Image.open('../images/'+img_file_name))
        img_links.append(img_file_name)



    f, axarr = plt.subplots(1, image_count_selected_patient, 
                            sharey=True,
                            figsize=(12, 4))
    f.suptitle(f'{image_count_selected_patient} Images for Patient ID: {selected_patient_id}', fontsize=14)

    counter_img = 0
    for index, row in selected_patient_df.iterrows():
        img = np.asarray(Image.open('../images/'+img_links[counter_img]))
        axarr[counter_img].imshow(img)
        axarr[counter_img].set_title("Laterality: "+str(row["laterality"])+"\n"+
                                    "View: "+str(row["view"])+"\n"+
                                    "Cancer: "+str(row["cancer"]))
        counter_img+=1


def plot_loss_accuracy_grid2_v2(results):
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    x_ticks_range = 1

    # epochs = range(len(results["train_loss"]))
    epochs = np.arange(1, len(results["train_loss"])+1)

    plt.figure(figsize=(14, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, x_ticks_range))
    plt.plot(epochs, train_loss, label="train_loss", color = "red")
    plt.plot(epochs, test_loss, label="test_loss", color = "blue")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)
    plt.grid(True,linestyle=':')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, x_ticks_range))
    plt.plot(epochs, train_accuracy, label="train_accuracy", color = "red")
    plt.plot(epochs, test_accuracy, label="test_accuracy", color = "blue")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True,linestyle=':')
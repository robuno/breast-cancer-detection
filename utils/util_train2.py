import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time
from datetime import datetime


def val_log_saver(model_results, train_test_opt):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y__%H_%M")

    f = open(date_time+"__"+train_test_opt+".txt", "w")

    for val in model_results[train_test_opt]:
        # print(val)
        f.write(str(val)+"\n")

    f.close()



def train_one_epoch2(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_func: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    # start_time_one_epoch = time.time()
    model.train()

    train_loss = 0
    train_acc = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_func(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad() # make zero grads to start fresh each forward pass
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    # end_time_one_epoch = time.time()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc




def test_one_epoch(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_func: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval() 

    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_func(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc





def train2(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_func: torch.nn.Module,
          epochs: int,
          device: torch.device,
          log_txt_saver: bool) -> Dict[str, List]:

    acc_loss_dict = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": [] }
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_one_epoch2(model=model,
                                          dataloader=train_dataloader,
                                          loss_func=loss_func,
                                          optimizer=optimizer,
                                          device=device)

        test_loss, test_acc = test_one_epoch(model=model,
                                          dataloader=test_dataloader,
                                          loss_func=loss_func,
                                          device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_accuracy: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_accuracy: {test_acc:.4f}"
        )

        acc_loss_dict["train_loss"].append(train_loss)
        acc_loss_dict["train_acc"].append(train_acc)
        acc_loss_dict["test_loss"].append(test_loss)
        acc_loss_dict["test_acc"].append(test_acc)

    if log_txt_saver == True:
        val_log_saver(acc_loss_dict,"train_loss")
        val_log_saver(acc_loss_dict,"train_acc")
        val_log_saver(acc_loss_dict,"test_loss")
        val_log_saver(acc_loss_dict,"test_acc")

    return acc_loss_dict

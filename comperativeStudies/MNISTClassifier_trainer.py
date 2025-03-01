import numpy as np
import pandas as pd

import torch


def classifier_train(classifier, train_loader, optimizer):
    classifier_losses = []
    classifier.train()

    losses = 0
    num_batches = 0

    for i, (X, y) in enumerate(train_loader):
        X = X.float()
        y = y.long()
        optimizer.zero_grad()
        classify = classifier(X)
        loss = torch.nn.functional.cross_entropy(classify, y)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        num_batches += 1

    avg_loss = losses / num_batches
    print(f'T Loss: {avg_loss:.4f}')
    classifier_losses.append(avg_loss)
    return classifier_losses

def classifier_val(classifier, val_loader):
    classifier.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.float()
            y = y.long().squeeze()
            classify = classifier(X)
            val_loss += torch.nn.functional.cross_entropy(classify, y).item()

    val_loss /= len(val_loader)
    print(f'Val Loss: {val_loss:.4f}')
    return val_loss



def gen_labels(classifier, state_dictionary, df_synth, batch_size):
    # checkpoint = torch.load(f'{state_dictionary}', map_location=torch.device("cpu"))
    # classifier.load_state_dict(checkpoint)
    classifier.eval()
    labels_list = []
    batch_size = batch_size
    df_synth = pd.read_csv(df_synth)
    df_synth = df_synth.apply(lambda col: col.str.strip("[]").astype(float) if col.dtype == "object" else col)
    X = torch.tensor(df_synth.values, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = X[i:i + batch_size]
            classify = classifier(X_b)
            sft = torch.nn.functional.softmax(classify, dim=1)
            pred = np.argmax(sft, axis=1)
            labels_list.append(pred)
        pseudo_labels = torch.cat(labels_list)

    return pseudo_labels


def classifier_test(classifier, state_dictionary, test_loader):
    # checkpoint = torch.load(f'{state_dictionary}', map_location=torch.device("cpu"))
    # classifier.load_state_dict(checkpoint)
    classifier.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            X = inputs.float()
            y = labels.long()
            classify = classifier(X)
            loss = torch.nn.functional.cross_entropy(classify, y)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f'Test Loss: {avg_loss:.4f}')

    return avg_loss

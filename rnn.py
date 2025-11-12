import torch
import torch.nn as nn
from torchtext import datasets
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim


class RNNCell(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, vocab_size, bias=False) # Waa
        self.linear2 = nn.Linear(vocab_size, vocab_size, bias=True) # Wax + ba
        self.tanh = nn.Tanh()

        self.final = nn.Linear(vocab_size, vocab_size, bias=True) # Wya + by

    def forward(self, x, previous_a):
        aa = self.linear1(previous_a)
        ax = self.linear2(x)
        x = torch.add(aa, ax)
        a = self.tanh(x)
        x = self.final(a)
        return x, a


class BasicNameDataset(Dataset):
    def __init__(self, file_path,transform=None, sos_key='<SOS>', eos_key='<EOS>', pad_key='<PAD>'):
        super().__init__()
        self.transform = transform
        self.file_path = file_path
        with open(file_path, 'r') as file:
            self.data = file.readlines()
        self.max_len = len(max(self.data, key=len))
        self.sos_key = sos_key
        self.eos_key = eos_key
        self.pad_key = pad_key
        self.alphabet = [self.sos_key] + list(set(''.join(self.data).lower())) + [self.eos_key, self.pad_key] 
        self.i2l = {idx: letter for idx, letter in enumerate(self.alphabet) }
        self.l2i = {letter: torch.tensor(idx) for idx, letter in enumerate(self.alphabet) }
        self.VOCAB_SIZE = len(self.l2i)
    
    def __getitem__(self, index):
        data = self.tokenize(self.data[index].lower()) # if self.transform  else self.data[index]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def tokenize(self, x):
        return torch.tensor([self.l2i[self.sos_key]] + [self.l2i[c.lower()] if c!='\n' else self.l2i[self.eos_key] for c in list(x)])

class TurkishNamesDataset(BasicNameDataset):
    def __init__(self, transform=None, sos_key='<SOS>', eos_key='<EOS>', pad_key='<PAD>'):
        file_path = "/Users/musazenbilci/Desktop/mosesopposite/andrej/makemore/tr_names.txt"
        super().__init__(file_path, transform, sos_key, eos_key, pad_key)

class DinoNamesDataset(BasicNameDataset):
    def __init__(self, transform=None, sos_key='<SOS>', eos_key='<EOS>', pad_key='<PAD>'):
        file_path = "/Users/musazenbilci/Desktop/mosesopposite/DeepLearningAI-RNN/W1A2/dinos.txt"
        super().__init__(file_path, transform, sos_key, eos_key, pad_key)

ddataset = TurkishNamesDataset()
alphabet = ddataset.alphabet
i2l = ddataset.i2l
l2i = ddataset.l2i
VOCAB_SIZE = ddataset.VOCAB_SIZE
torch.manual_seed(42)

make_one_hot = lambda x: torch.nn.functional.one_hot(x, VOCAB_SIZE).to(torch.float32)
make_zero_hot = lambda : torch.zeros(VOCAB_SIZE).to(torch.float32)

def pad_collate(batch):
    sequences = batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=l2i['<PAD>'])
    return padded_sequences

def cell_trainer():
    dloader = DataLoader(ddataset, batch_size=4, shuffle=True, collate_fn=pad_collate)


    initial_lr = 1e-2
    cell = RNNCell(VOCAB_SIZE)
    optimizer = optim.SGD(cell.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=5e-2)
    a0 = make_zero_hot()

    EPOCH=10
    for ep in range(EPOCH):
        epoch_loss = 0
        for batch in dloader:
            batch_loss = 0
            for word in batch:
                a_prev = a0
                loss = 0
                for ci in range(len(word)-1):
                    if word[ci] == l2i[ddataset.sos_key]:
                        continue
                    
                    optimizer.zero_grad()
                    onehot = make_one_hot(word[ci])
                    logits, act = cell(onehot, a_prev)
                    loss += criterion(logits, word[ci+1])
                    a_prev = act

                    if word[ci] == l2i[ddataset.eos_key]: # skip paddings
                        break
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            # print("BATCH_LOSS", batch_loss / 4)
            epoch_loss += batch_loss / 4

        print("EPOCH_LOSS", epoch_loss / 384)
        scheduler.step(epoch_loss / 384)
        if initial_lr != optimizer.param_groups[0]['lr']:
            print(f"LR Changed {optimizer.param_groups[0]['lr']}")
            initial_lr = optimizer.param_groups[0]['lr']
    return cell
        
def save_model(cell, name="turkish_rnn_cell.pth"):
    torch.save(cell.state_dict, f'./{name}')


def inference(cell):
    output = ''
    test_ind = l2i[ddataset.sos_key]
    input, a_prev = make_one_hot(torch.tensor(test_ind)), make_zero_hot()
    output += i2l[test_ind.item()]

    while True:
        logits, act = cell(input, a_prev)
        probs = torch.softmax(logits, 0)
        chosen = torch.argmax(probs)
        if chosen == l2i[ddataset.eos_key]:
            break
        output += i2l[chosen.item()]
        input = make_one_hot(chosen)
        a_prev = act
    return output

def main():
    cell = cell_trainer()
    out = inference(cell)
    print(out)


if __name__ == '__main__':
    main()
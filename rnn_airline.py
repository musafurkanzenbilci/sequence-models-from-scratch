import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def tokenize_month_info(input='1949-01'):
    year, month = input.split('-')
    diff = (int(year) - 1949)*12 + int(month)
    return torch.tensor(diff).to(torch.float32)

class AirlinePassengersDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        self.data = pd.read_csv(URL, index_col=False)
        self.scaler = MinMaxScaler()
        passengers_frame = self.data.Passengers.to_frame()
        self.scaler.fit(passengers_frame)
        scaled_passengers = self.scaler.transform(passengers_frame)
        self.passengers = scaled_passengers
        self.transform = transform

    
    def __getitem__(self, index):
        month, passenger = tokenize_month_info(self.data.Month[index]), torch.tensor(self.passengers[index]).to(torch.float32)
        return (month, passenger)
    
    def __len__(self):
        return len(self.data)


dataset = AirlinePassengersDataset()
split = int(len(dataset) * 0.9)
train_data, test_data = Subset(dataset, list(range(0, split))), Subset(dataset, list(range(split, len(dataset))))
loader = DataLoader(train_data, batch_size=4, shuffle=False)
test_loader = DataLoader(test_data, batch_size=len(dataset)-split, shuffle=False)


class RNNCell(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size, bias=False) # Waa
        self.linear2 = nn.Linear(input_size, input_size, bias=True) # Wax + ba
        self.tanh = nn.Tanh()

        self.final = nn.Linear(input_size, input_size, bias=True) # Wya + by

    def forward(self, x, previous_a): # also called hidden state
        aa = self.linear1(previous_a)
        ax = self.linear2(x)
        x = torch.add(aa, ax)
        a = self.tanh(x)
        x = self.final(a)
        return x, a


def train():
    initial_lr = 1e-1
    cell = RNNCell()
    optimizer = optim.SGD(cell.parameters(), lr=initial_lr)
    criterion = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, threshold=0.0005)
    a0 = torch.tensor([0]).to(torch.float32)

    EPOCH=1

    for ep in range(EPOCH):
        epoch_loss = 0
        for batch in loader:
            batch_loss = 0
            optimizer.zero_grad()
            a_prev = a0
            passengers = batch[1]
            
            if len(passengers)<2:
                continue

            for p in range(len(passengers) - 1):
                logits, a = cell(passengers[p], a_prev)
                loss = criterion(logits, torch.tensor([passengers[p+1]]))
                batch_loss += loss
                a_prev = a

            batch_loss.backward()
            optimizer.step()
            # print("Batch avg loss", batch_loss.item() / 4)
            epoch_loss += batch_loss.item() / 4

        print("EPOCH_LOSS",ep, epoch_loss / len(loader))
        scheduler.step(epoch_loss / len(loader))
        if initial_lr != optimizer.param_groups[0]['lr']:
            print(f"LR Changed {optimizer.param_groups[0]['lr']}")
            initial_lr = optimizer.param_groups[0]['lr']
    return cell

def test(cell):
    torch.set_grad_enabled(False)
    output = []
    labels = []
    input, a_prev = torch.tensor([test_loader.dataset[0][1]]), torch.tensor([0.]).to(torch.float32)

    for i in range(1,14):
        logits, act = cell(input, torch.tensor([a_prev]))
        output.append(dataset.scaler.inverse_transform(logits.reshape(-1, 1)))
        labels.append(dataset.scaler.inverse_transform(test_loader.dataset[i][1].reshape(-1, 1)))
        input = logits
        a_prev = act
    # print(output)
    # print(labels)
    torch.set_grad_enabled(True)
    return output, labels


def calculate_metrics(output, labels):
    toutput = torch.tensor(np.array(output))
    tlabels = torch.tensor(np.array(labels))
    # mape (mean absolute percentage error)
    # mean((actual-forecast) / actual)
    mape = torch.mean(torch.abs((torch.squeeze(toutput)-torch.squeeze(tlabels)) / torch.squeeze(tlabels))) # 0.1556
    print(f"mape {mape:.4}")

    # mae (mean absolute error)
    mae = torch.mean(torch.abs((torch.squeeze(toutput)-torch.squeeze(tlabels)))) # 68.1379
    print(f"mae {mae:.4}")

    # rmse (root mean square error)
    rmse = torch.sqrt(torch.mean(torch.square((torch.squeeze(toutput)-torch.squeeze(tlabels))))) # 80.0241
    print(f"rmse {rmse:.4}")
    return mape, mae, rmse

def main():
    cell = train()
    output, labels = test(cell)
    calculate_metrics(output, labels)



if __name__=='__main__':
    main()
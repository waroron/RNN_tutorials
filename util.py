import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os
import time
import numpy as np
from PIL import ImageChops, ImageStat
import math
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_passengers_data(path='datum/airline-passengers.csv', bd=100):
    data = pd.read_csv(path)
    passengers = data['Passengers'].values
    index = np.arange(0, len(passengers), 1)

    passengers = passengers.reshape(len(passengers), 1)
    index = index.reshape(len(index), 1)

    return (passengers[:bd], index[:bd]), (passengers[bd:], index[bd:])


class DiscreteDatasets(Dataset):
    def __init__(self, data):
        index = np.arange(0, len(data), 1)

        self.index = torch.from_numpy(index)
        self.data = torch.from_numpy(data)
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # if self.data_transform:
        #     noised_img = self.data_transform(self.df[i])
        # else:
        #     noised_img = self.df[i]
        return self.index[i], self.data[i]


class SimpleRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(n_input, n_hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x, h=None):
        output, hp = self.rnn(x.unsqueeze(1), h)
        output = self.out(output.squeeze(1))
        return output, hp

    def loss(self, x, z):
        y, hp = self.forward(x)
        return F.mse_loss(y, z)


def train(epoch, model, device="cuda"):
    model.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor()])
    print(device)
    train, test = get_passengers_data()

    model = model.to(device)
    optimizer = Adam(model.parameters())

    train_data = torch.from_numpy(np.array(train[0])).float().view(-1, 1)
    train_index = torch.from_numpy(np.array(train[1])).float()
    test_data = torch.from_numpy(test[0]).float()
    test_index = torch.from_numpy(test[1]).float()

    train = TensorDataset(train_index, train_data)
    train_loader = DataLoader(train, batch_size=40, shuffle=True)

    print('All datasets have been loaded.')

    for num in range(epoch + 1):
        total_loss = 0
        calc_time = 0
        test_loss = 0
        start = time.time()
        for batch_idx, (index, data) in enumerate(train_loader):
            image = index.to(device)
            label = data.to(device)

            optimizer.zero_grad()
            loss = model.loss(image, label)
            total_loss += float(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

            total_loss /= len(train_loader)
            # test_loss = evaluate(model, testloader, device)

        end = time.time()
        calc_time += (end - start)
        data_dict = {'num_epoch': num,
                     'training_loss': total_loss,
                     'test_loss': test_loss}
        print('Train Epoch: {} \tLoss: {:.6f} \tTest Loss: {:.6f} \tCalculation Time: {:.4f}sec'.format(
            num, total_loss, test_loss, calc_time))

        # denoising evaluation via testset
        # if num % 10 == 0:
        #     denoised_imgs, test_psnr = denoise_test(model, testloader, device)
        #     psnr_dict = {'test_psnr': test_psnr}
        #     data_dict.update(psnr_dict)
        #     append_csv_from_dict(experimental_dir, csv_name, data_dict)
        #     epoch_dir = os.path.join(experimental_dir, str(num))
        #     save_torch_model(epoch_dir, MODEL_PATH, model)
        #
        #     for num, denoised in enumerate(denoised_imgs):
        #         if num >= save_max:
        #             break
        #         denoised_name = os.path.join(epoch_dir, 'denoised_{}.png'.format(num))
        #         denoised.save(denoised_name)


if __name__ == '__main__':
    # train, test = get_passengers_data()
    model = SimpleRNN(1, 30, 1)
    train(10000, model)


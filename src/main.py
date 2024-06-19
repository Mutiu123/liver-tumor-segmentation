import torch
from dataPreprocess import prepprocess
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from train import train
from monai.losses import DiceLoss

# data and model directories
data_dir = 'C:/Users/adegb/Desktop/Computer Vision Projects/Liver-Tumor-Segmentation/dataset'
model_dir = 'C:/Users/adegb/Desktop/Computer Vision Projects/Liver-Tumor-Segmentation/model'
data_in = prepprocess(data_dir, cache=True)

num_epochs = 120

device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3, 
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, num_epochs, model_dir)

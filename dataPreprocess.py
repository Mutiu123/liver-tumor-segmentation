import numpy as np
import dicom2nifti
import nibabel as nib
import matplotlib.pyplot as plt
import os
from glob import glob
import shutil
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstD,
    Spacingd,
    ToTensord,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,

)
from monai.data import (
    Dataset, 
    DataLoader,
    CacheDataset,
)
from monai.utils import set_determinism, first



def prepprocess(input_data, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, batch_size=1, spatial_size=[128, 128, 64], cache=True):
  
    set_determinism(seed=42)

    train_volumes_input_path = sorted(glob(os.path.join(input_data, "TrainData", "*.nii.gz")))
    train_segmentation_input_path = sorted(glob(os.path.join(input_data, "TrainLabels", "*.nii.gz")))

    val_volumes_input_path = sorted(glob(os.path.join(input_data, "ValData", "*.nii.gz")))
    val_segmentation_input_path = sorted(glob(os.path.join(input_data, "ValLabels", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(train_volumes_input_path , train_segmentation_input_path)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(val_volumes_input_path, val_segmentation_input_path)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    if cache:
        train_set = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_set, batch_size=batch_size)

        test_set = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        return train_loader, test_loader

    else:
        train_set = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_set, batch_size=batch_size)

        test_set = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        return train_loader, test_loader



def show_patient(data, slice=1, train=True, test=False):

    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    
    if train:
        plt.figure("train model visualization ", (12, 8))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {slice}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, slice], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {slice}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, slice])
        plt.show()
    
    if test:
        plt.figure("test visualisation", (12, 8))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {slice}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, slice], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {slice}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, slice])
        plt.show()


def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val

import numpy as np
import torch.nn
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
from json import dumps


class ConvBlock(torch.nn.Module):
    '''CNN convolutional layer constructor'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        return x

class Network(torch.nn.Module):
    '''Full net with Conv and Dense layers and forward pass'''
    def __init__(self):
        super().__init__()
        
        self.conv = torch.nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=256),
        )
                
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 32),
            torch.nn.RReLU(0.1, 0.2),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 2),
        )
        

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = torch.sigmoid(self.fc(x))
        return x

transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
        ])

def get_json_results_for_n_rand_images(path=str(sys.argv[1]), size=100, model=Network(), transform=transform):
    """Returns json string of net predictions in format '{'female/00005.jpg': 'female'}.
        To run the script on some folder - pass FOLDER as first param: python process.py FOLDER.
        To specify the size of json result pass it as second param."""
    try:
        size = int(sys.argv[2])
    except IndexError:
        pass

    # choosing indices for 100 random images
    indices = np.random.randint(low=0, high=100001, size=size)
    # opening dir
    data = datasets.ImageFolder(path, transform=transform)
    # 100 images filenames
    filename = [file[0].split('/')[-2] + '/' + file[0].split('/')[-1] for file in np.array(data.imgs)[indices]]
    # images and their labels custom upload
    images, labels = [data[index][0] for index in indices], [data[index][1] for index in indices]
    images = torch.cat([img.float() for img in images], dim=0).reshape(size, 3, 64, 64)
    labels = torch.Tensor(labels)

    # prediction part
    pretrained_weights = f'weights_affine/weight_best29_affine.pt'  # Suppose last epoch model trained to be the best one
    model.load_state_dict(torch.load(pretrained_weights))
    preds = model.forward(images).detach().numpy()
    preds = np.argmax(preds, axis=1)

    # creating demanded objects for dict -> json format
    filename, gender = filename, np.where(preds, 'male', 'female')

    return dumps({img: label for img, label in zip(filename, gender)})

if __name__ == '__main__':
    print(sys.argv[0])
    res = get_json_results_for_n_rand_images()
    print(res)
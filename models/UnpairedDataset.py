import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(286, Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
}


class ImageFolder:
    def __init__(self, root):
        self.root = root
        self.paths = os.listdir(root)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        idx = idx % len(self)
        return os.path.join(self.root, self.paths[idx])
    

class UnpairedDataset(Dataset):
    def __init__(self, root, mode):
        """
        root must have trainA trainB testA testB as its subfolders
        mode must be either 'train' or 'test'
        """
        assert mode in 'train test'.split(), 'mode should be either train or test'
        
        super().__init__()
        pathA = os.path.join(root, mode+"A")
        self.dirA = ImageFolder(pathA)
        
        pathB = os.path.join(root, mode+"B")
        self.dirB = ImageFolder(pathB)
    
        self.transforms = data_transforms[mode]

        print(f'Found {len(self.dirA)} images of {mode}A and {len(self.dirB)} images of {mode}B')
        
    def __len__(self):
        return max(len(self.dirA), len(self.dirB))
    
    def load_image(self, path):
        image = Image.open(path)
        if self.transforms:
            image = self.transforms(image)
        return path, image
    
    def __getitem__(self, idx): #doesnt support slices, we dont want them
        # we use serial batching
        pathA, imgA = self.load_image(self.dirA[idx])
        pathB, imgB = self.load_image(self.dirB[idx])
        return {
            'A': imgA, 'pathA': pathA,
            'B': imgB, 'pathB': pathB
        }
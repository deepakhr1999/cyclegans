import os
import random
import base64
from PIL import Image
from io import BytesIO


class MonetDataset:
    def __init__(self):
        self.root = 'C:/Users/Deepak H R/Desktop/data/monet2photo/'
        self.testA = os.listdir(self.root + '/testA')
        self.testB = os.listdir(self.root + '/testB')
        self.idx = -1
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.testA)
        random.shuffle(self.testB)
    

    def load_image(self, path):
        buffered = BytesIO()
        image = Image.open(path)
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def __len__(self):
        return max(len(self.testA), len(self.testB))

    def __getitem__(self, idx):
        print("Index :", idx)
        path1 = self.root + 'testA/' + self.testA[idx % len(self.testA)]
        a = self.load_image(path1)

        path2 = self.root + 'testB/' + self.testB[idx % len(self.testB)]
        b = self.load_image(path2)
        return {
            'pathA': 'testA/' + self.testA[idx % len(self.testA)],
            'testA': a,
            'pathB': 'testB/' + self.testB[idx % len(self.testB)],
            'testB': b
        }

    def get_pair(self, command):
        if command == 'next':
            self.idx += 1
        else:
            self.idx -= 1
        return self[self.idx]
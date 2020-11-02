import os
import sys
import random
import urllib.parse
import shutil
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import DataLoader
from models.UnpairedDataset import UnpairedDataset
from models.CycleGan import CycleGan
from models.utils import load_weights
import numpy as np
import jinja2
from tqdm import tqdm

def render_template_to_string(template, output):
    with open(template, 'r') as file:
        s = file.read()

    with open('web/good_examples.txt', 'r') as file:
        good_examples = file.read().split()
    
    context = {
        'blocks' : [
            Item(basename)
            for basename in os.listdir('web/images/testA')
            if basename in good_examples
        ]
    }

    out = jinja2.Template(s).render(**context)
    with open(output, 'w') as file:
        file.write(out)


class Item:
    def __init__(self, basename):
        # not that page inside the web folder so we omit this in the path here
        self.testA = os.path.join('images/testA/', basename)
        self.fakeB = os.path.join('images/fakeB/', basename)
        self.basename = basename

class PageMaker:
    def __init__(self, root, ckpt=''):
        self.root = root

        test = UnpairedDataset(self.root, 'test')
        self.loader = iter(DataLoader(test, batch_size=5, shuffle=True))

        self.model = CycleGan()
        if ckpt != '':
            load_weights(ckpt, self.model)

        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.model.cuda()

    def tensor_to_img(self, x):# x is a 3D tensor output from the generator 
        if self.gpu:
            x = x.cpu()
        x = x.numpy()
        x = np.transpose(x, (1,2,0))
        x = (x + 1)*127.5
        x = x.astype('uint8')
        return Image.fromarray(x)
        # buffered = BytesIO()
        # image.save(buffered, format="JPEG")
        # return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def path_b64(self, path):
        buffered = BytesIO()
        image = Image.open(path)
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def load_images(self, pathA):
        return [self.path_b64(p) for p in pathA]

    def convert_output(self, fakeB): # fakeB is a 4D tensor output from generator
        return [self.tensor_b64(x) for x in fakeB]

    def __len__(self):
        return len(self.loader)

    def forward_batch(self, batch):
        """
            Makes a forward pass on the batch['A'] image
            saves the input image to web/images/testA
            saves the output image to web/images/fakeB under the same name
        """
        if self.gpu:
            x = batch['A'].cuda()
        else:
            x = batch['A']

        # forward pass
        with torch.no_grad():
            fakeB = self.model.genX(x)

        # store input images by copying to web/images/testA
        for pathA in batch['pathA']:
            base = os.path.basename(pathA)
            dst = os.path.join('web/images/testA/', base)
            shutil.copy(pathA, dst)

        # store output images by copying to web/images/fakeB
        for i, pathA in enumerate(batch['pathA']):
            base = os.path.basename(pathA)
            dst = os.path.join('web/images/fakeB/', base)
            self.tensor_to_img(fakeB[i]).save(dst)

    def forward(self):
        # make folders necessary
        os.makedirs('web/images/testA', exist_ok=True)
        os.makedirs('web/images/fakeB', exist_ok=True)

        for i, batch in enumerate(tqdm(self.loader)):
            # batch is a dict having keys A, B, pathA, pathB
            self.forward_batch(batch)


def main(args):
    if args.refresh:
        print("--refresh set to true. Running neural network on ")
        pageMaker = PageMaker(args.dataroot, args.ckpt)
        pageMaker.forward()

    #render to template
    render_template_to_string('web/template.html', 'web/index.html')
    url = 'file://' + os.path.abspath('web/index.html').replace('\\', '/').replace(' ', '%20')

    print(f"Webpage ready. Go to {url} at your browser!")
"""Testing models
    x = load(test_files/input.pt)
    y, z = load(test_files/output.pt)
    we test the following relation
    
    y = genX(x)
    z = disY(y)
    
    where torch seed is zero and nets are init using utils.init_weights
"""
import unittest
import torch
import os
from models import ResnetGenerator, PatchDiscriminator
from models.utils import init_weights


def get_nets():
    torch.manual_seed(0)
    genX = ResnetGenerator.get_generator()
    init_weights(genX)

    disY = PatchDiscriminator.get_model()
    init_weights(disY)

    return genX, disY

def get_tensors():
    dirname = os.path.dirname(__file__)
    filename = dirname + '/test_files/input.pt'
    
    x = torch.load(filename)

    filename = filename = dirname + '/test_files/output.pt'
    targetY, targetZ = torch.load(filename)
    return x, targetY, targetZ


x, targetY, targetZ = get_tensors()
genX, disY = get_nets()

class TestModel(unittest.TestCase):
    def test_generator(self):
        with torch.no_grad():
            y = genX(x)

        diff = ((y - targetY)**2).sum()
        
        self.assertAlmostEqual(diff.cpu().item(), 0.0)

    def test_discriminator(self):
        with torch.no_grad():
            z = disY(targetY)

        diff = ((z - targetZ)**2).sum()
        
        self.assertAlmostEqual(diff.cpu().item(), 0.0)


if __name__ == '__main__':
    unittest.main()


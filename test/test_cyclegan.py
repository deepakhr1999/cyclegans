"""
Assert that the CycleGan model is imported correctly
"""
import unittest
import torch
import os
from models.CycleGan import CycleGan
torch.manual_seed(0)

def numel(net):
    return sum(p.numel() for p in net.parameters())

net = CycleGan()

class TestCycleGan(unittest.TestCase):
    def test_cg_nparams(self):        
        nparams = numel(net)
        self.assertEqual(nparams, 10602436 * 2)
        
    def test_cg_output_shape(self):
        dirname = os.path.dirname(__file__)
        filename = dirname + '/test_files/input.pt'
        x = torch.load(filename)
        bsize = x.shape[0]

        with torch.no_grad():
            y1 = net.genX(x)
            self.assertEqual(y1.shape, (bsize, 3, 256, 256))

            y2 = net.genY(x)
            self.assertEqual(y2.shape, (bsize, 3, 256, 256))
            
            z1 = net.disY(y1)
            self.assertEqual(z1.shape, (bsize, 1, 30, 30))
            
            z2 = net.disX(y2)
            self.assertEqual(z2.shape, (bsize, 1, 30, 30))
        

if __name__ == '__main__':
    unittest.main()


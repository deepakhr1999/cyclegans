import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchsummary import summary
import itertools
import pytorch_lightning as pl


from models import PatchDiscriminator, ResnetGenerator
from models.utils import ImagePool, init_weights, set_requires_grad

class CycleGan(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # generator pair
        self.genX = ResnetGenerator.get_generator()
        self.genY = ResnetGenerator.get_generator()
        
        # discriminator pair
        self.disX = PatchDiscriminator.get_model()
        self.disY = PatchDiscriminator.get_model()
        
        self.lm = 10.0
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None

        for m in [self.genX, self.genY, self.disX, self.disY]:
            init_weights(m)
    
    def configure_optimizers(self):
        optG = Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        optD = Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)
            
    def generator_training_step(self, imgA, imgB):        
        """cycle images - using only generator nets"""
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)
        
        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)
        
        sameB = self.genX(imgB)
        sameA = self.genY(imgA)
        
        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')
        
        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')
        
        # compute extra losses
        identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)
        
        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)
        
        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss
        self.log('gen_loss', self.genLoss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        
        return self.genLoss
    
    def discriminator_training_step(self, imgA, imgB):
        """Update Discriminator"""        
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)
        
        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')
        
        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')
        
        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')
        
        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeA, 'fake')
        
        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        self.log('dis_loss', self.disLoss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return self.disLoss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgA, imgB = batch['A'], batch['B']
        discriminator_requires_grad = (optimizer_idx==1)
        set_requires_grad([self.disX, self.disY], discriminator_requires_grad)
        
        if optimizer_idx == 0:
            return self.generator_training_step(imgA, imgB)
        else:
            return self.discriminator_training_step(imgA, imgB)        


if __name__ == '__main__':
    model = CycleGan()
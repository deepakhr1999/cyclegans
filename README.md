# Cycle GAN on Pytorch Lightning
A Pytorch Lightning implementation of the research paper [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
Given two groups of images (we use Monet's paintings and a bunch of real images here), we want a function that takes an image from one domain and translates it to another domain. A vanilla generator maps a latent distribution to a desired output distribution. Instead of a vector sampled from a normal distribution, the cyclegan generator uses an image sampled from the input domain. The generator tries to fool the discriminator, a network that is being trained to classify if an image belongs to the target domain.

## Pytorch-Lightning
Pytorch Lightning simplifies the development of complex models. Personally, I find it easier to write code for new models in pytorch because it is easier to debug. The downside is that I end up spending too much time in writing training loops and checkpointing functionality. Enter [pytorch-lightning](https://www.pytorchlightning.ai/). All I have to do now is to write a function for calculating loss from a single training batch (precisely which is the novelty presented in the paper). Everything else, the boilerplate - checkpointing, progressbar and logging is already implemented.

## Prerequisites
```sh
# Pytorch
At <https://pytorch.org>, enter your preferences and run the command that shows up.
Training requires multiple gpus

# Pytorch Lightning
$ pip install pytorch-lightning

# Others
$ pip install torchsummary Pillow tqdm unittest flask
```

## Acknoledgements and Credits
##### Links
 - Cycle Gan [research paper](https://arxiv.org/abs/1703.10593)
 - pytorch-CycleGAN-and-pix2pix, [the original repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
 - Tensorflow [tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan) on cyclegan

To reduce model oscillation while training, the authors use the strategy from [Shrivastava et al.](https://arxiv.org/abs/1612.07828). A pool of 50 latest generated images is stored. An image sampled from this pool is set as input to the discriminator during training. The code for maintaining a pool of images is at [utils.py](../blob/main/models/utils.py) and was taken directly from this [file](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py) in the original repository. The weight initialization code has been taken from this [networks.py](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py).

##### Note
To verify my implementation of the network architecture, I cloned the [original pytorch cyclegan repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), sampled a random tensor input and made a forward pass on their generator and discriminator networks. These tensors have been saved [here](https://github.com/deepakhr1999/cyclegans/tree/main/test/test_files) folder as .pt files. I made a forward pass on the network I implemented and made sure that the output was the same. For reproducibility of network initialization, torch seed was set to zero.

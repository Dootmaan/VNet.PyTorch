# Reproduction of *V-Net* in PyTorch

This repo is created because I found that most repos on Github trying to reproduce V-Net by PyTorch have too many differences from the orginal paper *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image*, which may cause "potential misunderstandings". 

Here are some additional explanation:
- My work is mainly based on @zengyu714's code. The full network is constructed in a single file so you can easily adapt it to any of your projects.
- The network is now much closer to the one described in the paper(in my view), but still have one thing different -- normalization. This repo uses InstanceNorm3d between Conv3d and PReLU.

Please star this repo if you find it helpful, and if you find any other places of the codes seem different from the one in that paper, please inform me without hesitation. Thank you very much.

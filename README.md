# pytorch_universal_trainer

This repository contains a simple, easy to read and highly customizable distributed universal trainer in Pytorch.

## Introduction

Have you been in a situation where you wished you could easily make changes in the pre-exisiting Pytorch trainers in the wild (aka HuggingFace Trainer, PyTorch Lightening etc.) ?

Did you want to know more about how highly scalable distributed training works in PyTorch but were confused by 1000+ lines of complicated code?

Do you want to make the trainer highly customizable for your usage / model / metrics and still run smoothly on any cloud platform?

Have you run into Out of Memory issues or extremely slow training even when using fancy GPUs with extremely large models?

If the answer to the following questions is Yes, congratulations, you've come to the right place !!

The goal of this project is to make highly scalable (100s of GPUs) state of the art distributed trainer easily accessible, where the Pytorch user can dig deep, understand the inner workings,
and make changes in the future according to their own project needs. The best way to utilize this trainer is to clone, understand the training flow and make as many changes as
possible without restrictions. It serves as a starting point for your own custom trainer, which you can use for all Pytorch projects with minimal changes so that you can spend precious time
developing cool models / research tasks without worrying about the trainer / scaling the model to large datasets

Caution: this trainer is not for folks looking for black box trainers. There are plenty out there HuggingFace trainer, PyTorch Lightening to name a few. Using this code needs some experience
with PyTorch and the user should be ready to take a deep dive into the code and make custom changes. In my personal experience, black box trainers are too rigid at times, extremely complicated
to read and understand, not easy to make custom changes and may not work on your company's cloud platform. This is the major motivation for open-sourcing this work! 


## Getting Started

This section is aimed to describe the basic set of packages you would need. Note that providing a simple installation file that works for all cloud platforms is tricky. Thus it is left upto the
user to figure out the training enviroment. The code is tested using the following:

python 3.7.11
cuda 10.1
cudnn 7
torch 1.10.2+cu102
transformers 4.11.3
fairscale 0.4.6

The most important of them all is a torch, transformers and fairscale. These are absoultely necessary!

Apologies in advance for this. You might need to spend sometime setting up the environment, but once that is done, it is smooth sailing from there.

I wish I could provide more support on this but as long as you have a working cuda enviroment with pytorch, it should be straight-forward. One suggestion is to use a docker / aws image with pytorch
and cuda setup. fairscale is a relatively new library so there might be some version changes. Please stick to the one I've used or you might need to debug a few things. transformers library also might
have some version issues so please try the version above and get the code running before making custom changes. Unfortunately I have not used AWS to test this code, so wont be able to provide an image
for easy usage.

Todo in the future would be to provide a docker image / aws ami with the environment setup and ready to go!



```
{

}
```

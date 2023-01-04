# pytorch_universal_trainer

This repository contains a simple, easy to read and highly customizable distributed universal trainer in Pytorch.
The documentation is extremely verbose since it is meant as a tutorial as well! Please patiently read the whole thing
to learn the stuff quickly!!

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

**Caution:** this trainer is not for folks looking for black box trainers. There are plenty out there HuggingFace trainer, PyTorch Lightening to name a few. Using this code needs some experience
with PyTorch and the user should be ready to take a deep dive into the code and make custom changes. In my personal experience, black box trainers are too rigid at times, extremely complicated
to read and understand, not easy to make custom changes and may not work on your company's cloud platform. This is the major motivation for open-sourcing this work!

**Warning:** Your manager will be definitely surprised how you trained a model which takes days in hours / minutes. Please use the free time to enjoy the weather / take a stroll around a nearby park, 
or whatever that makes you happy ;)

With this trainer, I have been able to train on a 500GB dataset massive language modeling task under 10 days on 80+ GPUs, highly accelerating my training jobs!

## Features

This trainer supports the following:

1. **Simple PyTorch 1 node DataParallel:** In brief, this copies the model on one of the GPUs and uses the remaining GPUs for parallel training where the batches are splitted evenly across GPUs.
This option is not recommended as it is extremely slow compared to the ones below even on one node. (Refer PyToch doc for further reading)

2. **PyTorch DistributedDataParallel:** Recommended starting point for distributed training in PyTorch. In brief, this creates one process for each GPU available, the model is copied on all the GPUs
and the inter-process communication happens for training.

3. **Fairscale optimizations:** This supports model gradient and parameter sharding across GPUs. Extremely useful for large models like GPT-3. Highly recommend refering the official fairscale documentation
for more reading. Mainly used for extremely big models (500 million+ parameters). In most cases, option 2 should be fine.

Each of the above options supports **float-16 mixed precision training using Torch AMP** . This is highly useful to save GPU memory since it essentially allows almost 2x the batch size over full precision
float 32 training.

## Environment Configurations

This section is aimed to describe the basic set of packages you would need. Note that providing a simple installation file that works for all cloud platforms is tricky. Thus it is left upto the
user to figure out the training enviroment. The code is tested using the following:

**python 3.7.11**  
**cuda 10.1**  
**cudnn 7**  
**torch 1.10.2+cu102**  
**transformers 4.11.3**  
**fairscale 0.4.6**  

The most important of them all is a **torch, transformers and fairscale**. These are absoultely necessary!

Apologies in advance for this. You might need to spend sometime setting up the environment, but once that is done, it is smooth sailing from there.

I wish I could provide more support on this but as long as you have a working cuda enviroment with pytorch, it should be straight-forward. One suggestion is to use a docker / aws image with pytorch
and cuda setup. fairscale is a relatively new library so there might be some version changes. Please stick to the one I've used or you might need to debug a few things. transformers library also might
have some version issues so please try the version above and get the code running before making custom changes. Unfortunately I have not used AWS to test this code, so wont be able to provide an image
for easy usage.

A simple test for your environment is to try and import torch and see if it detects cuda by putting a simple tensor on the gpu device. Tru to import transformers and see if it throws any error.

Todo in the future would be to provide a docker image / aws ami with the environment setup and ready to go!

## Getting Started

As a test example, we will be training a simple binary classifier on the **Stanford SST-2 sentiment analysis** task. The model we will be using is the pre-trained **HuggingFace Facebook Bart Base model**.
The dataset is already provided in the data directory. You would need to download the model weights and the necessary files and put them in the **facebook/bart-base** directory. There is a **note** file
indicating the steps on where to download the exact model weights. In my experience, using from_pretrained() method doesnt work well in a distributed environment where the exact paths are needed.

All the files are extremely readable with comments. **There is a specific comment with ----->** which indicates all the custom changes that the end user would make for their respective tasks. I highly
recommend to spend sometime to read the code carefully following the -----> arrow for future tasks. For the above task, this is not needed.

Below indicates a brief description of the files:

**main.py:** Starting point for your training code, calls the respective trainer class and the dataloader class. Start reading the code from here and follow the trace.

**data_loader.py:** Initializes the pytorch dataloader and does batch pre-processing like tokenzation / padding etc.

**transformers_train.py:** The actual trainer code which includes a training and evaluate functions.

**parameters.yml:** The parameters file needed for different training parameters.

Before jumping into the training commands, we need to make sure the directories and the paths are setup. In the parameters.yml file, make sure the paths for the **data_dir, pretrained_model_dir, model_dir**
is setup correctly. By default the code assumes that the **pytorch_universal_trainer** directory is in your home (~/) directory and the paths are with respect to that. Without this step, there will be errors
for sure. The model_dir will be created to write the weights if one doesn't exist already!

## Jumping into Distributed Training

I highly recommend reading the various options in the **parameters.yml** file first, as it provides a detailed description of the various parameters the trainer supports. The steps below assume that the directories
and environment is setup and ready to go.

###### Single node PyTorch DistributedDataParallel (DDP) training

We will first get started to see if we can train on a single node in distributed fashion. If this succeeds, scaling to multiple nodes wont be a big issue. The flags needed for this are:  

**mode:** There are 2 modes 'train' and 'test'. We will set the 'train' mode first.  
**continue_training:** Since this is a fresh training job, this will be False.  
**is_distributed:** True  
**use_native_fp16:** True  

Mixed precision training is optional but I recommend as a standard default for all distributed training jobs.

The most important flags are in the dist-params section of the parameters.yml file. 

**nodes:** Since we are using one node, set this to 1.  
**gpus:** Depending on the node, set the number of GPUs available. Usually 8 for a typical node. But set accordingly.  
**current_node:** This is the index of the node. For one node, it is 0.  
**master_addr:** Extremely important to set the correct ip of the node. ifconfig -a and copy the inet address.  
**master_port:** Set any available port.  

With the params set correctly in parameters.yml file, the command to run is simple as follows. Note this command stays the same for all training tasks (more details on multi-node training later)

```
{

python main.py --params=parameters.yml

}

```

After the following command, you should see the trainer begin. In the model directory, there will be a logger_logs directory. Note that since PyTorch DDP creates mutliple processes for each GPU, there will be
log files corresponding to the number of GPUs. Use tail -f training_0.log, to check one of the logs to see it is printing the loss. If not, the processes are stuck and something is wrong. By default the trainer logs
loss depending on the log steps ratio set, and saves the model checkpoint after every epoch and does evaluation on the dev set! The trainer also writes tensorboard logs for visualization later!

Once the training is successful, you should see **3 checkpoint files for the 3 epochs**. Looking at **logger_logs/training_0.log**, you should see the best eval checkpoint step. Once you have that checkpoint number, we are ready
to use the trainer in **'test' mode** for testing on the test set!

In **main.py on line 127**, you need to insert the correct checkpoint number based on your best checkpoint. That is all the changes that are needed in this file. 

Jump to the **parameters.yml** file, and change the mode to **'test'**. Use the above command again, the same one used for training. After it is finished, you should see **dev_results.json and test_results.json** in the model directory!
This indicates the accuracy metric for the dev and test sets respectively!

If you see the files, congratulations! You have successfully trained a model in distributed fashion on one node using multiple GPUs!!

###### Multi-node PyTorch DistributedDataParallel (DDP) training

**Warning: Proceed to this step only if the single node training above works successfully!**

The multi-node training involves a few more steps and shouldn't be complicated. For multiple nodes, ideally you should have a shared file system accessible to all the nodes and the paths should be set accordingly in parameters.yml file.
If you dont have access to the shared file system, you would need to replicate the exact directory structure with the data on other nodes (essentially copying the whole directory with data and pretrained weights). Without this, PyTorch
wont be able to find the neccessary files and the processes will be stuck or throw an error.

Once the directory structures are setup, remember from the previous example that we set some **dist-params**. This will be very crucial now! Lets dive deeper into them. For simplicity **start with 2 nodes first**. The process is the same for
more nodes! Note that now since there are 2 nodes, the parameters.yml file will be present on 2 nodes.

1. **Step 1:** Out of the 2 nodes, pick one of them as the master node and note the ip of that node. The other node will be the child node communicating with it. There can only be **one master node for 'n' nodes**

2. **Step 2:** In dist-params, the **'nodes'** parameter will now be 2, since we are using 2 nodes. **'gpus'** params indicates number of gpus on each node, should be the same on each node, ideally 8 each. This param
stays the same on both parameters.yml file on both the nodes.

3. **Step 3:** On the first master node, set the **'current_node'** index to 0. This indicates its the first node. For the **second node**, in the parameters.yml file on the second node, set this to 1.

4. **Step 4:** Set the **master_addr** from the master node and **'master_port'** values the same in both files on different nodes. This tells Pytorch these processes belong together!

Once the above is set, we need to run the same command as above on both the nodes one after the other. The pytorch process will wait until the other node processes also joins!

That's it! Thats all there is, now the processes will communicate between nodes and hopefully in the model directory you should have the corresponding logs and checkpoints. I internally use torch.multiprocessing.spawn to launch the
processes, so hopefully things should run smoothly.

There is a chance that this might not work at first, dont fret, it is relatively easy to debug since PyTorch throws verbal errors which are easy to find fixes for online. Running it first time would be tricky but hopefully things will
workout!

If it works out and you are wondering if I use 10+ nodes, am I supposed to start processes manually on each node, it is a very valid question. The answer is no. You would need to write a bash script that starts the processes for you!
It gets non-trivial when you are using kubernetes or a large clsuter which dynamically assigns ip for each nodes. In these cases, the best advice I can give you is to consult the respective platform team for advice!

###### Multi-node PyTorch Fairscale optimizations training

If you have reached this far, many congratulations! It only gets easier from here on! This option is useful only for extremely large models (Bert-Large, GPT-3 500 million+ parameters etc.) For most usecases, DDP should be fine. But if
you still want to explore this option (why not :)), read below!

With the above parameters.yml file, it is extremely easy to switch from DDP to Fairscale! In addition to the flags set in the above step, you only need
the following:

1. **For Optimizer Sharding Only:** Only set the **use_fairscale_optimizer_sharding: True**. Note since this is still distributed training, the is_distributed flag and the use_native_fp16 flag will still be set to True like in DDP.

2. **For the full Optimizer, Gradient, Model Parameter Sharding FSDP:** Since this shards everything, we need to set the **use_fairscale_optimizer_sharding: False** and need to set these three flags to True: **use_fairscale_model_sharding,
use_fairscale_autowrap, use_fairscale_reshard_forward**

That is all there is to the fairscale optimizations. It closely follows DDP and everything else remains the same, except the flags which change!

## Customizing the trainer for your tasks and general large scale training tips

If you have reached this far, it means you are fairly comfortable with the trainer code and know the specifics of distributed training. This section is for guidance on how to customize this trainer for your own tasks!

The trainer is designed to be re-usable for multiple different tasks with only a few necesary changes. In order to do that, carefully follow the **comments with ----->**, starting from the main.py file. Ideally only the
data_loader, tokenizer and the model should change. For the evaluation loop, you can include your own metrics as well. In my opinion, all the changes you would need are marked with the **----->** comments. Other specifics
about the trainer should be task agnostic! I expect you to take full freedom with the trainer code and make changes as necessary for your task, assuming you have read the code and are comfortable with the logical flow of
the code!

For extremely large datasets, I highly recommend checking out the [Huggingface Datasets library](https://huggingface.co/docs/datasets/). For 100+ GBs of training data, it is important to have an efficient data reader and in
my experience, no library does it as easily as this.

When you scale the model to 100s of GPUs, you will also come across a lot of training instabilities, NaN losses etc. Optimizer params become very crucial for large batch sizes when scaling on 100s of GPUs. Large Batch training
is definitely not trivial and I highly recommend reading into the Adam Optimizer specifics or try to use the default params from research papers as much as possible. 

## Note on pull requests

Warning, I wont be accepting pull requests for cosmetic changes to the code. Its upto the user to fully customize this code for their needs.

I will only be accepting pull requests for major bug fixes or new features (deepspeed integration) that excite me! This trainer may not be perfect and I would humbly appreciate any new cool additions to it to improve my own
training workflows as well (surprise me!)

## Acknowledgements

This section is to appreciate all the inspirations behind this trainer! Without it, it wouldn't be possible at all to write this trainer!

[Huggingface Transformers](https://huggingface.co/)

I personally cannot thank folks at Huggingface enough to open source their awesome trainer! It is the most feature complete trainer I have come across. I thoroughly read through every single line to understand what is going on,
and some best practices / syntax! My inspiration was to make the trainer more customizable and simple for my needs, helping me learn in the process as well :) 

[Fairscale](https://fairscale.readthedocs.io/en/latest/)

The guys at PyTorch never fail to surprise with their awesome work making extremely complicated concepts so easy to use in practice! Highly recommend reading their blogs and articles. Recently they even trained a GPT-3 using fairscale
which goes to show how powerful their library is!

[Awesome DDP Tutorial by Kevin Kaichuang Yang](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)

This tutorial is what led me to understand PyTorch DDP and clarify the basic concepts and get the simple DDP code running! Highly recommend to check the blog out if you are stuck somewhere!



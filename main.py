import argparse
import yaml
import logging
import os
import json

### -----> Pro tip: for larger datasets, look into huggingface's awesome datasets library, without that, it will be very painful to train on 150-500 gb dataset size

def read_files(filename):
    import pandas as pd
    data = pd.read_csv(filename, sep = '\t', header = None)
    titles = list(data[0])
    labels = list(data[1])

    return titles, labels

def data_flow(params):

    from data_loader import transformer_dataloader  ## need to import data_loader file 
    from transformers import BartTokenizerFast   ### -----> Make changes here for your choice of tokenizer
    import torch
    from torch.utils.tensorboard import SummaryWriter

    ## lines 23-39 are for initializing the logger and the tensorflow summary writer, no changes needed by the user
    
    # Initialize the logger
    if params['transformers-params']['mode'] == 'train':
        if params['transformers-params']['is_distributed']:
            log_file = 'training_' + str(torch.distributed.get_rank()) + '.log'
        else:
            log_file = 'training.log'
        writer = SummaryWriter(os.path.join(params['transformers-params']['model_dir'], "tensorboard_logs"))

    else:
        log_file = 'testing.log'
        writer = None
        
    log_path = os.path.join(params['transformers-params']['model_dir'], "logger_logs", log_file)
    logging.basicConfig(filename=log_path, filemode='w', level = logging.INFO)
    logger = logging.getLogger(__name__)
    
    ## logger initialization done !!
    
    ## load the pretrained tokenizer which is needed by the data loader
    
    pretrain_path = params['transformers-params']['pretrained_model_dir']
    tokenizer = BartTokenizerFast.from_pretrained(pretrain_path) ### -----> Make changes here for your your choice of tokenizer
    
    train_dataloader = None
    dev_dataloader = None
    test_dataloader = None

    # Read data and initialize the data loader
    # Depending on the task, the user needs to write their own read_files function and might need to change the data loader
    
    if params['transformers-params']['mode'] == 'train':
        
        train_titles, train_labels = read_files(os.path.join(params['data-params']['data_dir'], 'sentiment-train')) ### -----> Make changes here for the filename / read function
        train_dataloader = transformer_dataloader(train_titles, train_labels, params, tokenizer, 'train')
        
        if params['data-params']['is_dev_file']:
            dev_titles, dev_labels = read_files(os.path.join(params['data-params']['data_dir'], 'sentiment-dev')) ### -----> Make changes here for the filename / read function
            dev_dataloader = transformer_dataloader(dev_titles, dev_labels, params, tokenizer, 'dev')
        
    else:
        
        dev_titles, dev_labels = read_files(os.path.join(params['data-params']['data_dir'], 'sentiment-dev')) ### -----> Make changes here for the filename / read function
        dev_dataloader = transformer_dataloader(dev_titles, dev_labels, params, tokenizer, 'dev')
        
        
        test_titles, test_labels = read_files(os.path.join(params['data-params']['data_dir'], 'sentiment-test')) ### -----> Make changes here for the filename / read function
        test_dataloader = transformer_dataloader(test_titles, test_labels, params, tokenizer, 'test')

    logger.info("  Done reading data into dataloaders ...")

    return train_dataloader, dev_dataloader, test_dataloader, tokenizer, logger, params, writer


def transformers_flow(gpu, params):

    from transformers import BartForSequenceClassification, BartConfig ### -----> Make changes here for your model
    import transformers_train as tr
    import torch
    import random
    import numpy as np
    import time

    if type(params) == str:
        params = yaml.load(params, Loader = yaml.FullLoader)
    
    ## This section initializes the torch process group. Remember in Torch distributed, each GPU will have its own process. If you are using 3 nodes with 8 GPUS each, there will be 24 different processes. That is the world size, number of different processes. rank is what is the index of the current process. So with 3 nodes and 24 GPUs, rank will go from 0 to 23. gpu variable indicates with respect to the local node, what is the gpu index. so for each node above index will go from 0 to 7.
    
    if params['transformers-params']['is_distributed']:
        rank = params['dist-params']['current_node'] * params['dist-params']['gpus'] + gpu
        params['dist-params']['current_rank'] = rank
        params['dist-params']['current_gpu'] = gpu
        torch.distributed.init_process_group(backend='nccl',
                                                 world_size=params['dist-params']['world_size'], 
                                                 rank=rank)
       
    train_dataloader, dev_dataloader, test_dataloader, tokenizer, logger, params, writer = data_flow(params)

    # setting seed is important. The user should set their own seed!
    seed = 42 ### -----> Make changes here if you want to set a different seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if params['transformers-params']['mode'] == 'train':

        start_time = time.time()
        
        best_step = tr.transformers_train(params, logger, train_dataloader, dev_dataloader, tokenizer, writer)

        logger.info("  Done training ...")
        logger.info("  The best step for the whole training is %s", best_step) ### Will only be returned obviously if eval file is passed
        total_time = time.time() - start_time
        logger.info('\n')
        logger.info("  The whole training took %s seconds for %s epochs", total_time, params['transformers-params']['num_epochs'])
        writer.close()

    else:

        best_step = "396" ### -----> Make changes here. insert checkpoint value here after model is trained!
        model_dir = os.path.join(params['transformers-params']['model_dir'], 'checkpoint-{}'.format(best_step))
        
        ### -----> Make changes here for your specific model. I like to load the original model from config file and then load the trained weights after
        config = BartConfig.from_pretrained(params['transformers-params']['pretrained_model_dir'])
        config.num_labels = params['transformers-params']['num_labels']
        model = BartForSequenceClassification(config)
        model.load_state_dict(torch.load(model_dir)["model_state_dict"])
        
        if not params['transformers-params']['is_distributed']:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda", params['dist-params']['current_gpu'])

        model.to(device)
        tr.transformers_evaluate(params, model, dev_dataloader, write_results = True, data_type = 'dev')
        tr.transformers_evaluate(params, model, test_dataloader, write_results = True, data_type = 'test')

def dist_transformers_flow(params):
    
    ## gpus is numner of gpus per node, nodes is number of server nodes you are using!
    world_size = params['dist-params']['gpus'] * params['dist-params']['nodes']
    params['dist-params']['world_size'] = world_size
    os.environ['MASTER_ADDR'] = params['dist-params']['master_addr'] ## extremely important, there will only be one master node controlling all the other processess in different nodes
    os.environ['MASTER_PORT'] = params['dist-params']['master_port'] ## port could be any usable one!
    
    import torch.multiprocessing as mp
    
    mp.spawn(transformers_flow, nprocs=params['dist-params']['gpus'], args=(params,))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="parameters.yml file to train and evaluate the spell correction models!")
    args = parser.parse_args()
    params = yaml.load(open(args.params), Loader = yaml.FullLoader)
    
    model_dir = params['transformers-params']['model_dir']
    log_dir = os.path.join(model_dir, "logger_logs")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

   
    if not params['transformers-params']['is_distributed']:
        transformers_flow(0, params)
    else:
        dist_transformers_flow(params)
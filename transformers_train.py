import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast


from transformers.trainer_pt_utils import distributed_concat
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data.distributed import DistributedSampler

import fairscale
from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.nn.wrap import auto_wrap
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler

from transformers import BartConfig ### -----> Make changes here for your model imports
from transformers import BartForSequenceClassification  ### -----> Make changes here for your model imports
from transformers.optimization import AdamW, get_linear_schedule_with_warmup  ### -----> Make changes here for your choice of optimizer imports

import json
from tqdm import tqdm, trange
import os

from sklearn.metrics import accuracy_score  ### -----> Make changes here for your own choice of metrics

def transformers_result(trues, predictions):
    results = {
        "Accuracy": accuracy_score(y_true = trues, y_pred = predictions)
    }
    return results

def create_optimizer(params, model, num_train_optimization_steps): ### -----> Make changes here for your model/task
    
    adam_epsilon = 1e-8 #1e-6 ### -----> For extremely large datasets, when using a very large batch size on 8-10 nodes, be careful with this parameter, often results in NaN loss due to gradient explosion with Adam

    warmup_proportion = params['transformers-params']['warmup_proportion']
    
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    ## manual weight decay is not recommended with Adam
    optimizer_kwargs = {'lr':float(params['transformers-params']['learning_rate']), 'eps':adam_epsilon}
    optimizer_cls = AdamW
    
    if params['transformers-params']['use_fairscale_optimizer_sharding']:
        optimizer = OSS(params=optimizer_grouped_parameters, optim=optimizer_cls, 
                        broadcast_fp16 = params['transformers-params']['use_native_fp16'], 
                        **optimizer_kwargs)
    else:
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    warmup_steps = int(num_train_optimization_steps * warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
    
    return optimizer, scheduler

def transformers_train(params, logger, dataloader, eval_dataloader = None, tokenizer = None, writer = None):

    ## device and number of GPUs, if not is_distributed, it will use torch.nn.DataParallel as default based on number of GPUs
    if not params['transformers-params']['is_distributed']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(" Device found %s", device)
        n_gpu = torch.cuda.device_count()
        logger.info(" Number of GPUs found %s", n_gpu)
    else:
        device = torch.device("cuda", params['dist-params']['current_gpu'])
        logger.info("  Device found %s", device)
        n_gpu = 1  ## since torch distributed launches individual processes for each gpu, n_gpu is 1
    
    if params['transformers-params']['is_distributed']:
        ## this is needed by fairscale fsdp, where gather operation must happen on the same device!
        torch.cuda.set_device(device)
    
    if params['transformers-params']['continue_training']: ## if continuing training from previous checkpoint, need to load the dict on the correct device
        # continue training path would be the exact last checkpoint number to resume training from!
        state_dict = torch.load(params['transformers-params']['continue_training_path'], map_location=device)

    ## load the model
    ## This time for classification, I need to also load the config file to change the number of labels
    ## This example below is only for a simple classification task, depending on your task, load the correct model!
    
    config = BartConfig.from_pretrained(params['transformers-params']['pretrained_model_dir']) ### -----> Make changes here for your model
    config.num_labels = params['transformers-params']['num_labels'] ### -----> Make changes here for your model
    model = BartForSequenceClassification.from_pretrained(params['transformers-params']['pretrained_model_dir'], ### -----> Make changes here for your model
                                                         config = config)
    
    ## load the previous model state dict if continuing training
    if params['transformers-params']['continue_training']:
        model.load_state_dict(state_dict['model_state_dict'])
    
    logger.info("  Model weights initialized ...")
    
    ## put model on device
    ## This is very important to do before the optimizer initialization, since optimizer uses model.named_parameters which also have to be
    ## on the same device!!
    model.to(device)

    ## some default params for training
    max_grad_norm = 1.0

    ## logging, saving step defaults
    save_steps = int(len(dataloader) / params['transformers-params']['gradient_accumulation_steps'])
    logging_steps = int(save_steps / params['transformers-params']['log_steps_ratio'])
    save_steps = int(save_steps / params['transformers-params']['save_steps_ratio'])

    if params['transformers-params']['continue_training']:
        num_train_optimization_steps = state_dict['num_train_optimization_steps']

    else:
        num_train_optimization_steps = int(len(dataloader) / params['transformers-params']['gradient_accumulation_steps']) * params['transformers-params']['num_epochs']
    
    logger.info('  Len dataloader: %s', len(dataloader))
    logger.info('  Number of optimization steps: %s', num_train_optimization_steps)
    
    ## create optimizer and scheduler
    ## with fairscale zero3 and autowrap, optimizer is initialized later
    if not params['transformers-params']['use_fairscale_model_sharding']:
        optimizer, scheduler = create_optimizer(params, model, num_train_optimization_steps)
        
        if params['transformers-params']['continue_training']: ## if continuing training from a previous checkpoint, need to load previous optimizer states too!
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
        logger.info("  Optimizer is initialized ...")
    
    
    ## this section is for float16 mixed precision training, which needs a scaler to scale / unscale weights. For Sharded Models, we need a different special scaler
    scaler = None
    if params['transformers-params']['use_native_fp16']:
        if params['transformers-params']['use_fairscale_optimizer_sharding'] or params['transformers-params']['use_fairscale_model_sharding']:
            scaler = ShardedGradScaler()
        else:
            scaler = torch.cuda.amp.GradScaler()
        
        if params['transformers-params']['continue_training']:
            scaler.load_state_dict(state_dict['scaler_state_dict'])

    if not params['transformers-params']['is_distributed']: ## vanilla pytorch single node DataParallel
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
    else:

        if params['transformers-params']['use_fairscale_optimizer_sharding']: ## fairscale optimizer sharding code
            model = ShardedDDP(model, optimizer)

        elif params['transformers-params']['use_fairscale_model_sharding']: ## fairscale model sharding code

            mixed_precision = params['transformers-params']['use_native_fp16']
            reshard_after_forward = params['transformers-params']['use_fairscale_reshard_forward']

            if params['transformers-params']['use_fairscale_autowrap']:
                model = auto_wrap(model)

            model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=reshard_after_forward
                )
            
            ## Now create the optimizer for zero3!!
            optimizer, scheduler = create_optimizer(params, model, num_train_optimization_steps)
            
            if params['transformers-params']['continue_training']:
                optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                scheduler.load_state_dict(state_dict['scheduler_state_dict'])

            logger.info("  Optimizer is initialized ...")
            
        else:
            model = torch.nn.parallel.DistributedDataParallel(  ## vanilla pytorch Distributed, no model / optimizer sharding
                    model,
                    device_ids=[params['dist-params']['current_gpu']],
                    output_device=params['dist-params']['current_gpu'],
                    find_unused_parameters=(
                        not getattr(model.config, "gradient_checkpointing", False)
                        if isinstance(model, PreTrainedModel)
                        else True
                    ),
                )
        
    model.zero_grad()
    logger.info("  Model put on different GPUs ...")

    ## training loop begins
    
    if params['transformers-params']['continue_training']:
        global_step = state_dict['global_step']
        tr_loss = state_dict['tr_loss']
        logging_loss = state_dict['logging_loss']
        epoch_loss = state_dict['epoch_loss']
        best_eval_metric = state_dict['best_eval_metric']
        best_step = state_dict['best_step']
    
    else:
        global_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0 ### -----> Make changes here for your model/task if you want to log different metrics
        best_eval_metric, best_step = 0.0, 0.0 ### -----> Make changes here for your model/task if you want to log different metrics
    
    train_iterator = trange(int(params['transformers-params']['num_epochs']), desc="Epoch", disable=False)

    logger.info("  Training loop is about to begin ...")
    logger.info(" \n")
    
    for epoch in train_iterator:

        if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)

        for step, batch in enumerate(epoch_iterator):
            
            # means of debugging only. Need to make sure training begins sucessfully in distributed mode and processes are not stuck
            if step == 0:
                logger.info("  Inside the for loop at step %s at epoch %s", step, epoch)
            
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch  ### -----> Make changes here according to the input your model needs. Note Bert also returns token type ids but for bart they are not needed
            
            if params['transformers-params']['use_native_fp16']:
                with autocast():
                    output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, return_dict=False)
                    loss = output[0] ### -----> Make changes here according to what your forward pass of the model returns. First return parameter should be loss
            else:
                output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, return_dict=False)
                loss = output[0] ### -----> Make changes here according to what your forward pass of the model returns. First return parameter should be loss
            
            # means of debugging
            if step == 0:
                logger.info("  Loss is computed successfully first...")

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
                ## do not need to mean the loss for distributed because loss is computed locally but only gradients are gathered

            if params['transformers-params']['gradient_accumulation_steps'] > 1:
                loss = loss / params['transformers-params']['gradient_accumulation_steps']
            
            if params['transformers-params']['use_native_fp16']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % params['transformers-params']['gradient_accumulation_steps'] == 0:

                if params['transformers-params']['use_native_fp16']:
                    scaler.unscale_(optimizer)
                
                ## gradient clipping section
                if hasattr(optimizer, "clip_grad_norm"):
                    # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                    optimizer.clip_grad_norm(max_grad_norm)
                elif hasattr(model, "clip_grad_norm_"):
                    # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                    model.clip_grad_norm_(max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                ## optimizer update section
                optimizer_was_run = True
                if params['transformers-params']['use_native_fp16']:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after

                else:
                    optimizer.step()

                ## the scheduler throws warning of lr_scheduler being called before optimizer, thats why this is needed!
                if optimizer_was_run:
                    scheduler.step()
                    
                ## add loss value to SummaryWriter object
                writer.add_scalar('Train Loss', loss.item(), global_step) ### logging the loss in tf writer is important
                
                ### -----> It will be upto you to log anything else, I only log essential items

                model.zero_grad(set_to_none=True)
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    if params['transformers-params']['is_distributed']:
                        logger.info("  Logging Loss : %s at step %s at rank %s\n\n", (tr_loss - logging_loss) / logging_steps, global_step,         torch.distributed.get_rank()) ### There will be a logger file for each gpu in distributed mode!
                    else:
                        logger.info("  Logging Loss : %s at step %s \n\n", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                if save_steps > 0 and global_step % save_steps == 0: 
                    if not params['transformers-params']['is_distributed']:
                        save_model(model, optimizer, scheduler, scaler, params, 
                                   global_step, num_train_optimization_steps, 
                                   tr_loss, logging_loss, epoch_loss, 
                                   best_eval_metric, best_step, logger)
                    else:
                        ## When using OSS / FSDP, the model needs to saved on all ranks according to the documentation!!
                        if params['transformers-params']['use_fairscale_model_sharding'] or params['transformers-params']['use_fairscale_optimizer_sharding']:
                            save_model(model, optimizer, scheduler, scaler, params, 
                                       global_step, num_train_optimization_steps, tr_loss, logging_loss, 
                                       epoch_loss, best_eval_metric, best_step, logger)

                        elif torch.distributed.get_rank() == 0:
                            save_model(model, optimizer, scheduler, scaler, params,
                                       global_step, num_train_optimization_steps, tr_loss, logging_loss, 
                                       epoch_loss, best_eval_metric, best_step, logger)

                    ## whenever I am saving the model, also run a eval loop to see how the model is performing
                    ### -----> Make changes here according to whether you want to evaulate or not. If there is no evalfile, the evaldataloader should be None
                    if eval_dataloader:
                        eval_metric = transformers_evaluate(params, model, eval_dataloader)
                        logger.info("  Global eval epoch metric: %s", eval_metric)
                        if eval_metric > best_eval_metric:
                            best_eval_metric = eval_metric
                            best_step = global_step
                            logger.info("  Found new best Global eval metric %s at %s steps", best_eval_metric, best_step)
                        logger.info('\n')
                    
                # break the loop because now there's no concept of epoch but we still need to stop training 
                # when remaining global steps are completed
                if params['transformers-params']['continue_training']:
                    if global_step == num_train_optimization_steps:
                        stop_training = True
                        break
                    else:
                        stop_training = False

        #### the inner for loop ends / epoch ends ####

        logger.info('\n')
        if params['transformers-params']['is_distributed']:
            logger.info("  Training epoch Loss: %s at step %s at rank %s ", (tr_loss - epoch_loss) / (step+1) * params['transformers-params']['gradient_accumulation_steps'], global_step, torch.distributed.get_rank())
        else:
            logger.info("  Training epoch Loss: %s at step %s ", (tr_loss - epoch_loss) / (step+1) * params['transformers-params']['gradient_accumulation_steps'], global_step)
        logger.info('\n')
        
        epoch_loss = tr_loss

        if eval_dataloader:  ### -----> Make changes here according to whether you want to evaulate or not.
            eval_metric = transformers_evaluate(params, model, eval_dataloader)
            logger.info("  Global eval epoch loss: %s", eval_metric)
            if eval_metric > best_eval_metric:
                best_eval_metric = eval_metric
                best_step = global_step
                logger.info("  Found new best Global eval loss %s at %s steps", best_eval_metric, best_step)
            logger.info('\n')
        
        ## break the main outerloop since training is now done, since global steps have been reached,
        ## even though while continuing training, we re-initialize the data loader.
        if params['transformers-params']['continue_training']:
            if stop_training:
                break


    #### The main outer training for loop ends ####

    if not params['transformers-params']['is_distributed']:
        save_model(model, optimizer, scheduler, scaler, params, 
                   global_step, num_train_optimization_steps, 
                   tr_loss, logging_loss, epoch_loss, 
                   best_eval_metric, best_step, logger)
    else:
        ## When using OSS / FSDP, the model needs to saved on all ranks according to the documentation!!
        if params['transformers-params']['use_fairscale_model_sharding'] or params['transformers-params']['use_fairscale_optimizer_sharding']:
            save_model(model, optimizer, scheduler, scaler, params, 
                       global_step, num_train_optimization_steps, tr_loss, logging_loss, 
                       epoch_loss, best_eval_metric, best_step, logger)

        elif torch.distributed.get_rank() == 0:
            save_model(model, optimizer, scheduler, scaler, params,
                       global_step, num_train_optimization_steps, tr_loss, logging_loss, 
                       epoch_loss, best_eval_metric, best_step, logger)

    return best_step


### -----> Make changes here if you want to save something else. I only save essentials needed for continuing training from a checkpoint for failures!
def save_model(model, optimizer, scheduler, scaler, params, 
               global_step, num_train_optimization_steps, 
               tr_loss, logging_loss, epoch_loss, 
               best_eval_metric, best_step, logger):

    output_dir = os.path.join(params['transformers-params']['model_dir'], 'checkpoint-{}'.format(global_step))
    
    save_dic = {}
    
    ## model sharding does unwrapping internally when collecting state_dict so dont need to do that!
    if not params['transformers-params']['use_fairscale_model_sharding']:
        model_to_save = model.module if hasattr(model, 'module') else model
        save_dic = {'model_state_dict': model_to_save.state_dict()}
    else:
        save_dic = {'model_state_dict': model.state_dict()}

    if params['transformers-params']['use_fairscale_optimizer_sharding']:
        # need to use all_ranks due to sharded optimizer
        optimizer.consolidate_state_dict(-1) ## this stupid -1 is important! All ranks should have the state. Refer docs!
        save_dic['optimizer_state_dict'] = optimizer.state_dict()
    else:
        save_dic['optimizer_state_dict'] = optimizer.state_dict()

    save_dic['scheduler_state_dict'] = scheduler.state_dict()
    if scaler:
        save_dic['scaler_state_dict'] = scaler.state_dict()
    save_dic['global_step'] = global_step
    save_dic['num_train_optimization_steps'] = num_train_optimization_steps
    save_dic['tr_loss'] = tr_loss
    save_dic['logging_loss'] = logging_loss
    save_dic['epoch_loss'] = epoch_loss
    save_dic['best_eval_metric'] = best_eval_metric
    save_dic['best_step'] = best_step
    
    ## The function can be called on all processes, but saving the final dict is needed on main process, breaks FSDP saving/loading
    if params['transformers-params']['is_distributed']:
        if torch.distributed.get_rank() == 0:
            torch.save(save_dic, output_dir)
    else:
        torch.save(save_dic, output_dir)

    logger.info('\n')
    logger.info("  Saving model checkpoint to %s", output_dir)
    logger.info('\n')


### -----> The function where most changes are needed, depending on the model output and the metrics you are using to evaulate!
def transformers_evaluate(params, model, eval_dataloader, write_results = False, data_type=None):

    if not params['transformers-params']['is_distributed']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda", params['dist-params']['current_gpu'])
    
    softmax = torch.nn.Softmax(dim=1)  ### -----> Make changes if it is not a classification task
    
    all_preds_tensor = None
    all_labels_tensor = None
    
    num_total_examples = len(eval_dataloader.dataset)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, attention_mask, labels = batch ### -----> Make changes here according to the input your model needs.
            
            if params['transformers-params']['use_native_fp16']:
                with autocast():
                    output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                    logits = output[1] ### -----> Make changes here according to the output of your model. My model returns logits for classification as the second parameter
            else:
                output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                logits = output[1] ### -----> Make changes here according to the output of your model. My model returns logits for classification as the second parameter
            
            logits = logits.detach()
            labels = labels.detach()
            
            if all_preds_tensor == None:
                all_preds_tensor = logits
            else:
                all_preds_tensor = torch.cat((all_preds_tensor, logits), dim=0)
            
            if all_labels_tensor == None:
                all_labels_tensor = labels
            else:
                all_labels_tensor = torch.cat((all_labels_tensor, labels), dim=0)

    print()
    print ("Distributed Concat started ...")
    print()
    ## need to concat the various tensors from across GPUs and nodes in distributed mode!!
    if params['transformers-params']['is_distributed']:
        all_preds_tensor = distributed_concat(all_preds_tensor, num_total_examples)
        all_labels_tensor = distributed_concat(all_labels_tensor, num_total_examples)

    print()
    print("Distributed Concat done")
    print()
    # softmax calculated
    all_preds_tensor = softmax(all_preds_tensor) ### -----> Make changes here if not a classification task
    preds_list = torch.argmax(all_preds_tensor, dim=1).tolist() ### -----> Make changes here if not a classification task
    labels_list = all_labels_tensor.tolist()
    

    results = transformers_result(labels_list, preds_list) ### -----> Make changes here if you are using a different metric function
    primary_metric = results["Accuracy"]
    
    if write_results:
        results_path = os.path.join(params['transformers-params']['model_dir'], data_type + '_results.json')
        prediction_path = os.path.join(params['transformers-params']['model_dir'], data_type + '_prediction.txt')

        with open(results_path, 'w') as f: ### -----> Make changes here depending on whether the results are returned as a json or not
            json.dump(results, f)

        fp = open(prediction_path, 'w') ### -----> Make changes here depending on what you want to write
        for item in preds_list:
            fp.write(str(item))
            fp.write('\n')
        fp.close()

    return primary_metric
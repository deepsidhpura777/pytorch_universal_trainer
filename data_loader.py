from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import SequentialDistributedSampler
from functools import partial
import torch


## Note this dataloader is specifically written for the Facebook Bart Model.
## If you have a different model. you would need to make sure the inputs and outputs are as expected.
## Alternatively you can also use HF's tokenizer to prepare the batch. This is an example of doing it manually

## collate function is extremely useful to do batch pre-processing on the fly! For NLP it would be tokenization / padding etc.

def transformer_bart_collate(batch, tokenizer, params): ### -----> Make changes here for your model/task

    sentences, labels = zip(*batch)
    sentences = list(sentences)
    labels = list(labels)
    
    if params['transformers-params']['use_max_length']:
        inputs = prepare_bart_batch(sentences, labels, tokenizer, params['transformers-params']['max_length'])
    else:
        inputs = prepare_bart_batch(sentences, labels, tokenizer, None)
    
    return (inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    
def return_tokens(string, tokenizer): ### -----> Make changes here for your model/task
    tokens = [tokenizer.bos_token] + tokenizer.tokenize(string) + [tokenizer.eos_token]
    return tokens, len(tokens)

def prepare_bart_batch(sentences, labels, tokenizer, max_length=None): ### -----> Make changes here for your model/task
    
    token_function = partial(return_tokens, tokenizer = tokenizer)

    sentence_tokens, sentence_lens = zip(*list(map(token_function, sentences)))
    
    sentence_tokens = list(sentence_tokens)
    sentence_lens = list(sentence_lens)

    ms = max(sentence_lens)

    if max_length:
        max_length = min(ms, max_length)
    else:
        max_length = ms
    
    attention_mask = []

    for i in range(len(sentence_tokens)):

        attention_mask.append([1] * len(sentence_tokens[i]))
        
        if len(sentence_tokens[i]) < max_length:
            sentence_tokens[i] = sentence_tokens[i] + [tokenizer.pad_token] * (max_length - len(sentence_tokens[i]))
            attention_mask[i] = attention_mask[i] + [0] * (max_length - len(attention_mask[i]))
        else:
            sentence_tokens[i] = sentence_tokens[i][:max_length]
            sentence_tokens[i][-1] = tokenizer.eos_token ## while truncating the eos token is lost at the end
            attention_mask[i] = attention_mask[i][:max_length]

    ## convert tokens to ids
    sentence_ids = list(map(tokenizer.convert_tokens_to_ids, sentence_tokens))
    
    ## convert to torch tensors and return the input dictionary, like the huggingface prepareseq2seq function.
    inputs = {'input_ids' : torch.tensor(sentence_ids), 'attention_mask' : torch.tensor(attention_mask), 'labels' : torch.tensor(labels)}

    return inputs


def transformer_dataloader(sentences, labels, params = None, tokenizer = None, data_type = None):
    
    zip_data = list(zip(sentences, labels))

    # Configure the sampler
    if data_type == 'train':
        if not params['transformers-params']['is_distributed']:
            sampler = RandomSampler(zip_data)
        else:
            sampler = DistributedSampler(zip_data, num_replicas=params['dist-params']['world_size'], 
                                         rank=params['dist-params']['current_rank'])
    else:
        if not params['transformers-params']['is_distributed']:
            sampler = SequentialSampler(zip_data)
        else:
            sampler = SequentialDistributedSampler(zip_data, num_replicas=params['dist-params']['world_size'], 
                                         rank=params['dist-params']['current_rank'])

    # Return the corresponding data loader
    
    # This is in non-distributed mode where batch size is the overall batch size. For example Batch size of 64 on 8 GPUs would mean 8 per GPU. This is for NON-Distributed Mode only !! In Distributed mode, each batch size is replicated on every GPU available !!
    if not params['transformers-params']['is_distributed']:
        batch_size = int(params['transformers-params']['batch_size'] / params['transformers-params']['gradient_accumulation_steps'])
    else:
        batch_size = params['transformers-params']['batch_size']

    ## trying out the bart model
    dataloader = DataLoader(zip_data, sampler = sampler, 
                            batch_size = batch_size, 
                            collate_fn = partial(transformer_bart_collate, tokenizer = tokenizer, params = params),
                            num_workers = params['transformers-params']['num_loader_workers'])
    
    return dataloader
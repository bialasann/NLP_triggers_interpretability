import torch 
import numpy as np
from tqdm import tqdm 
import time

from torch.utils.data import TensorDataset


# CONSTANTS
trigger_ids_neg = torch.tensor([1795, 14419, 21067, 24800, 24800, 27912, 12108, 25576, 15889, 11203,
                                8225,  8225,  8225,  8536,  1852,  1194, 17207,  8225,  1194,  8225,
                                8225, 13119,  1103,  1103, 26789, 10587, 11057, 24913,  4965,  1762 ])

trigger_ids_pos = torch.tensor([17038, 17103, 17748, 24862, 24862, 23463, 17216, 16777, 22847, 23959,
                                1103, 26559, 11116, 11116, 13831, 22677, 21876,  1103,  8558, 11116,
                                11116, 11116, 11624, 21755, 21518,  6034, 17735,  1103,  8558, 17216 ])

extracted_grads = []

def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])
    print(f'Len of extracted grads: {len(extracted_grads)}')

# Done a bunch of tests to figure out which embedding layer to take (probably this is the one)
"""
Returns the embedding matrix of dimensions (embedding_size, embedding_dimension)
Here embedding_size = vocab_size 
Both values can be obtained from get_vocab_size_embed_dim()
"""
def get_embedding_weight(language_model, vocab_size):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
          if module.weight.shape[0] == vocab_size: # only add a hook to wordpiece embeddings, not position embeddings
            return module.weight.detach()

# add hooks for embeddings
"""
Hooks are registered for each nn.Module object and are triggered by the forward or backward pass calls.
https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
"""
def add_hooks(language_model, vocab_size, hook_func):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == vocab_size: # only add a hook to wordpiece embeddings, not position
              print()
              print("Added a hook")
              module.weight.requires_grad = True
              hook = module.register_backward_hook(hook_func)
    return hook  
                                
# tokenizes sentences and concatenates piecewise array outputs into a single tensor
# BERT uses 3 arrays; does this for all three and packs into a TensorDataset object 
def tokenizer_function(input_data, labels, tokenizer = None, label_filter = None, flip_labels = False):
  input_ids = []
  attention_masks = []

  # pad / truncate reviews to a constant length
  # add special tokens to the start and end
  # differentiate real tokens from padding tokens with attention mask
  for sent in input_data:
    this_encoding = tokenizer.encode_plus(sent, truncation=True, padding='max_length', max_length = 512, return_attention_mask = True, return_tensors = 'pt')
    input_ids.append(this_encoding['input_ids'])
    attention_masks.append(this_encoding['attention_mask'])

  # concatenate tokens and prepare arrays for all sentences in the dataset
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)

  # filters out positive/negative reviews and returns only those input sentences
  if label_filter != None:
    filter_mask = [labels == label_filter]
    input_ids = input_ids[filter_mask]
    attention_masks = attention_masks[filter_mask]
    labels = labels[filter_mask]
  
  if flip_labels:
    print(f'Flipping labels from {label_filter} to {1-label_filter}')
    labels = 1-labels

  tokenized_data = TensorDataset(input_ids, attention_masks, labels)
  
  return tokenized_data


def append_triggers(batch, trigger_token_ids, device = 'cuda'):
    """
      - appends the triggers to the input sequence
      - ensures that the first token is the required first token CLS for DistilBERT 
      - adjusts the mask for this change
      - ensures that the shapes do not change

    number of sentences = batch_size = 32
    input_ids: dim = (batch_size, max_sequence_len) = (batch_size, 512) because DistilBERT handles a max sequence length of 512
    Unused token spaces among the 512 are padded
    start_token: pick out start token (CLS) from all sentences, dim = (batch_size,1)
    sequence: - all tokens except start token
              - truncates tokens = trigger_length from the end to accommodate trigger   
    b_input_ids_avd: concatenates start_token + trigger + sequence
    b_input_mask_avd: in DistilBERT's attention mask, 1 is paying attention, 0 is padding
              - we concatenate a string of 1's for the trigger to the existing mask
    Doc: https://huggingface.co/docs/transformers/glossary#attention-mask   
    """
    # get input data
    b_input_ids = batch[0]
    b_input_mask = batch[1]
    b_labels = batch[2]

    batch_size = b_input_ids.shape[0]
    trigger_ids_pos = trigger_token_ids.pos
    trigger_ids_neg = trigger_token_ids.neg

    assert len(trigger_ids_pos) == len(trigger_ids_neg), f"Positive and negative triggers are not the same length, {len(trigger_ids_pos)} and {len(trigger_ids_neg)} respectively."

    # context is trigger repeated batch size
    trigger_len = trigger_ids_pos.shape[0]
    trigger_tensor = trigger_ids_pos.unsqueeze(0).repeat(batch_size, 1)
    trigger_tensor[b_labels==0] = trigger_ids_neg
  
    # append trigger to the input 
    start_token = b_input_ids[:, :1]
    sequence = b_input_ids[:, 1:-trigger_len]
    sequence[sequence[:, -1] != 0, -1] = 102
    b_input_ids_avd = torch.cat((start_token, trigger_tensor, sequence), dim=1)
  
    # we need to extend the attention tensor 
    b_input_mask_avd = torch.cat((torch.ones(trigger_tensor.shape), b_input_mask[:, :-trigger_len]), dim=1)

    # labels to device
    b_labels = b_labels

    return b_input_ids_avd, b_input_mask_avd, b_labels

# get vocabulary size and embedding dimension
def get_vocab_size_embed_dim(model):
  for module in model.modules():
    if isinstance(module, torch.nn.Embedding):
      return module.weight.shape[0], module.weight.shape[1]

# gets the loss of the target_tokens using the triggers as the context
def get_model_output(language_model, batch, trigger_token_ids, output = None, device = 'cuda', eval = True, return_input = False):

  # append triggers to inputs
  if trigger_token_ids is None:
    b_input_ids, b_input_mask, b_labels = batch[0], batch[1], batch[2]
  else:  
    b_input_ids, b_input_mask, b_labels = append_triggers(batch, trigger_token_ids, device = device)
  
  b_input_ids = b_input_ids.to(device)
  b_input_mask = b_input_mask.to(device)
  b_labels = b_labels.to(device)

  language_model.zero_grad()
  if eval:
    with torch.no_grad():
        outputs = language_model(b_input_ids, 
                              attention_mask=b_input_mask, 
                              labels=b_labels)
  else:
      outputs = language_model(b_input_ids, 
                              attention_mask=b_input_mask, 
                              labels=b_labels)
    
  if output == 'loss':
    # loss is of type float (it is a single number)
    outputs = outputs.loss
  elif output == 'acc':
    logits = outputs.logits
    pred = logits.max(1).indices
    outputs = torch.mean(1 * pred == batch[2].to(device), dtype = torch.float)
  elif output == 'full':
    outputs = outputs

  
  if return_input:
    return outputs, b_input_ids
  else:
    return outputs

# evaluate model by computing logits and accuracy for a given data loader
# set label_filter to None if you want full dataset

def _evaluate_model(model, data_loader, label_filter = None, trigger_tokens = None, device = None):
    logits = []
    labels = []
    for i, batch in enumerate(data_loader):
      model.zero_grad()  
      
      # evaluate accuracy for the given trigger
      model_outputs = get_model_output(model, batch, trigger_tokens, device = device)
      logits.append(model_outputs.logits.detach().cpu())
      labels.append(batch[2].detach().cpu())

    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    pred = logits.argmax(axis = 1)

    return {
        "true_0" : 100 -np.mean(labels)*100, 
        "true_1" : np.mean(labels)*100, 
        "pred_0" : 100 -np.mean(pred)*100,
        "pred_1" : np.mean(pred)*100, 
        "acc"    : np.sum(pred == labels) / len(labels)
      }

def evaluate_model(model, data_loader, label_filter = None, trigger_tokens = None, device = None):
    print(f'The number of batches in the data loader is: {len(data_loader)}')

    dict1 = _evaluate_model(model, data_loader, label_filter = label_filter, trigger_tokens = None, device = device)

    print("Evaluate model - no triggers")
    print(f"Label 0: {dict1['true_0']} \d Pred 0: {dict1['pred_0']}")
    print(f"Label 1: {dict1['true_1']} \d Pred 0: {dict1['pred_1']}")
    print(f"Accuracy: {dict1['acc']}")

    if trigger_tokens is not None:
      dict1 = _evaluate_model(model, data_loader, label_filter = label_filter, trigger_tokens = trigger_tokens, device = device)
      print()
      print("Evaluate model - triggers")
      print(f"Label 0: {dict1['true_0']} \d Pred 0: {dict1['pred_0']}")
      print(f"Label 1: {dict1['true_1']} \d Pred 0: {dict1['pred_1']}")
      print(f"Accuracy: {dict1['acc']}")



# TODO: rewrite only used for finetuning
# model evaluation on test data

def fine_tune(model, train_loader, val_loader, optimizer, device = None):
    
    total_t0 = time.time()
    
    for epoch in tqdm(range(10)):
      total_train_loss = 0
      model.train()
      
      for step, batch in enumerate(train_loader):
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()  
        outputs = model(b_input_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # if(step %10 == 0):
        #   print(loss)

      avg_train_loss = total_train_loss / len(train_loader)   
      print("")
      print("Average training loss: {0:.2f}".format(avg_train_loss))
          
      print("")
      print("Running Validation...")

      t0 = time.time()

      # put the model in evaluation mode -
      model.eval()

      # tracking variables 
      total_eval_accuracy = 0
      total_eval_loss = 0
      nb_eval_steps = 0

        # evaluate data for one epoch
      for batch in val_loader:
          #
          # `batch` contains three pytorch tensors:
          #   [0]: input ids 
          #   [1]: attention masks
          #   [2]: labels 
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)

          with torch.no_grad():        

              outputs = model(b_input_ids, 
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
              
          # accumulate the validation loss.
          loss = outputs.loss
          logits = outputs.logits
          total_eval_loss += loss.item()

          # move logits and labels to CPU
          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()

          # calculate the accuracy for this batch of test sentences, and
          # accumulate it over all batches.
          pred = logits.max(1).indices
          classification_accuracy = torch.mean(1 * pred == batch[2].to(device), dtype = torch.float)
          total_eval_accuracy += classification_accuracy

      # report the final accuracy for this validation run
      avg_val_accuracy = total_eval_accuracy / len(val_loader)
      print("Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

      # calculate the average loss over all of the batches
      avg_val_loss = total_eval_loss / len(val_loader)
      
      # measure how long the validation run took
      validation_time = format_time(time.time() - t0)
      
      print("Validation Loss: {0:.2f}".format(avg_val_loss))
      print("Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")

    print("Total training took {:}".format(format_time(time.time()-total_t0)))

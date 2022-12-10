import torch 
import numpy as np

from torch.utils.data import TensorDataset


# CONSTANTS
trigger_ids_neg = torch.tensor([1795, 14419, 21067, 24800, 24800, 27912, 12108, 25576, 15889, 11203,
                                8225,  8225,  8225,  8536,  1852,  1194, 17207,  8225,  1194,  8225,
                                8225, 13119,  1103,  1103, 26789, 10587, 11057, 24913,  4965,  1762 ])

trigger_ids_pos = torch.tensor([17038, 17103, 17748, 24862, 24862, 23463, 17216, 16777, 22847, 23959,
                                1103, 26559, 11116, 11116, 13831, 22677, 21876,  1103,  8558, 11116,
                                11116, 11116, 11624, 21755, 21518,  6034, 17735,  1103,  8558, 17216 ])
                                
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

    # context is trigger repeated batch size
    trigger_len = trigger_token_ids.shape[0]
    trigger_tensor = trigger_token_ids.unsqueeze(0).repeat(batch_size, 1)
  
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

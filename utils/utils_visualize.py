import utils_model 
import utils_visualize 

import numpy as np
import matplotlib.pyplot as plt

import torch as torch
from torch.utils.data import DataLoader

def get_review_len(input_ids):
  review_len = (input_ids == 102).nonzero(as_tuple=True)[1].item()
  return review_len

def create_iterator(dataset, batch_size = 1, shuffle = False):
  input_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  input_iterator = iter(input_loader)

  return input_iterator

def resize(att_mat, end_idx=None, start_idx=0):
  """Normalize attention matrices and reshape as necessary."""
  att_res = []
  for _, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      print("Expended dimentions")
      att = np.expand_dims(att, axis=0)
    if end_idx is not None:
      att = att[:, :, start_idx:end_idx, start_idx:end_idx]
    att_res.append(att)
  att_res = np.stack(att_res)
  return att_res

def show_attention_per_layer(
    model_output, 
    input_ids, 
    max_len = 90,
    layers = [], 
    heads = list(range(12)),
    figsize=(30,5), 
    tokenizer = None, 
    n_heads = 12, 
    print_text = False, 
    cmap = plt.cm.coolwarm, 
    filename = None ):
  max_len = get_review_len(input_ids)
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  atten_mtx = torch.stack(model_output[2]).detach().cpu()
  atten = resize(atten_mtx, max_len)

  fig, axs = plt.subplots(len(layers), len(heads)+1, figsize = figsize)

  for i, head in enumerate(heads):
    for j, layer in enumerate(layers):
      axs[j][len(heads)].axis('off')
      im = axs[j][i].matshow(atten[layer][0][head], cmap=cmap)
      if print_text:
        axs[j][i].set_xticks(range(max_len))
        axs[j][i].set_xticklabels(tokens[:max_len], rotation=60)
        axs[j][i].set_yticks(range(max_len))
        axs[j][i].set_yticklabels(tokens[:max_len])

  # plt.suptitle('Attention heatmap by layer and head')
  plt.tight_layout()

  # cbar = axs[j][i].figure.colorbar(im, 
  #                         shrink=0.5 )

  cbar = fig.colorbar(im, ax=axs[:, len(heads)], shrink = 0.6, location='left' )

  if filename is not None:
    plt.savefig(filename, format='svg')

  plt.show()

class DatasetDictionary:
  def __init__(self, dataset, tokenizer):
    self.size = len(dataset)
    self.tokenizer = tokenizer
    self.dictionary = {}
    self.dictionary = self._create_dictionary(dataset)
    self.trigger_dict = {}

  def _create_dictionary(self, dataset): 
    reviews = dataset[:][0]
    unique_vals, unique_counts = reviews.unique(return_counts=True)

    dictionary = dict(zip(unique_vals.detach().cpu().numpy(), unique_counts.detach().cpu().numpy()))
    return dictionary 

  def get_count_by_ids(self, ids): 
    ds_count = list(map(self.dictionary.get, ids))
    ds_count = np.array([0 if x is None else x for x in ds_count])
    trigger_count = list(map(self.trigger_dict.get, ids))
    trigger_count = np.array([0 if x is None else x for x in trigger_count])
    count = ds_count + trigger_count
    return count

  def get_count_by_tokens(self, tokens): 
    ids = self.tokenizer.convert_tokens_to_ids(tokens)
    count = self.get_count_by_ids(ids)
    return count   

  def add_triggers(self, triggers):
    if triggers is None:
      trigger_dict = None
    else: 
      unique_vals, unique_counts = triggers.unique(return_counts=True)
      dictionary = dict(zip(unique_vals.detach().cpu().numpy(), unique_counts.detach().cpu().numpy() * self.size))
      self.trigger_dict = dictionary

def get_most_attented(attention_out, input_ids, n_words=5, layers = [5], tokenizer = None):
  attention_out = torch.stack(attention_out).detach().cpu().numpy()
  tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))

  most_attented_tokens_all = []
  most_attented_ids_all = []
  for head in range(12):
    for layer in layers:
      attention_score = (-(attention_out[layer][0][head]).sum(0))
      sorted_idsx = attention_score.argsort()
      most_attented_ids = sorted_idsx[:n_words] 
      most_attented_tokens = tokens[most_attented_ids]
      most_attented_tokens_all.append(most_attented_tokens)
      most_attented_ids_all.append(most_attented_ids)

  return most_attented_tokens_all, most_attented_ids_all

def get_most_attented_ds(input_iterator, model, triggers, tokenizer = None):
  most_attented_tokens = []
  for input in input_iterator:
    model_output, input_ids = utils_model.get_model_output(model, input, triggers, output = 'full', device = 'cuda', eval=True, return_input = True)

    attention_out = model_output[2]
    tokens, ids = get_most_attented(attention_out, input_ids, n_words=5, tokenizer = tokenizer)
    most_attented_tokens.append(tokens)
    
  most_attented_tokens = np.array(most_attented_tokens).flatten()
  tokens_unique, counts = np.unique(most_attented_tokens, return_counts=True)
  sorted_counts = (-counts).argsort()

  return tokens_unique[sorted_counts], counts[sorted_counts]

def quantify_attention(id_to, model, review, triggers=None):

  model_output, input_ids = utils_model.get_model_output(model, review, triggers, output = 'full', device = 'cuda', eval=True, return_input = True)

  max_len = get_review_len(input_ids) +1
  atten_mtx = torch.stack(model_output[2]).detach().cpu()
  atten = utils_visualize.resize(atten_mtx, max_len)

  idx_to = np.where(input_ids[0].detach().cpu()==id_to)[0]
  if len(idx_to) == 0:
    result = None
  else: 
    atten_by_id = atten.sum(axis=3).squeeze(axis=1)
    atten_by_id = atten_by_id[:, :, idx_to]

    atten_sum = atten_by_id.sum(axis=2)/len(idx_to)
    result = atten_sum/max_len
  
  return result

def quantify_attention_ds(id_to, dataset, model, triggers, tokenizer=None):
  # Without triggers
  n_occurences = 0
  atten_no_triggers = None
  input_iterator = create_iterator(dataset, batch_size=1, shuffle=False)
  for input in input_iterator:
    atten = quantify_attention(id_to, model, input, triggers=None)
    if atten is not None:
      n_occurences += 1
      if atten_no_triggers is None:
        atten_no_triggers = atten
      else:
        atten_no_triggers += atten
  atten_no_triggers = None if n_occurences == 0 else atten_no_triggers/n_occurences
  
  # With triggers
  n_occurences_2 = 0
  atten_triggers = None
  input_iterator = create_iterator(dataset, batch_size=1, shuffle=False)
  for input in input_iterator:
    atten = quantify_attention(id_to, model, input, triggers=triggers)
    if atten is not None:
      n_occurences_2 += 1
      if atten_triggers is None:
        atten_triggers = atten
      else:
        atten_triggers += atten
  atten_triggers = None if n_occurences_2 == 0 else atten_triggers/n_occurences_2
  
  return atten_no_triggers, n_occurences, atten_triggers, n_occurences_2

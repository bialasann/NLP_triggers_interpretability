import utils_model 

import torch as torch
import numpy as np
import matplotlib.pyplot as plt

def get_review_len(input_ids, len_treshold=90):
  max_len = min((input_ids == 102).nonzero(as_tuple=True)[1].item(), len_treshold) +1
  return max_len

def resize(att_mat, max_length=None):
  """Normalize attention matrices and reshape as necessary."""
  att_res = []
  for _, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      print("Expended dimentions")
      att = np.expand_dims(att, axis=0)
    if max_length is not None:
      att = att[:, :, :max_length, :max_length]
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
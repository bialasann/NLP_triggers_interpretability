import utils_model 

import torch as torch
import numpy as np

def show_attention_per_layer(model_output, input_ids, layer, figsize=(30,90), tokenizer = None):
  max_len = min((input_ids == 102).nonzero(as_tuple=True)[1].item(), 90) +1
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  atten_mtx = torch.stack(model_output[2]).detach().cpu()
  atten = resize(atten_mtx, max_len)
  
  fig, axs = plt.subplots(6,2, figsize = figsize)
  axs = np.ravel(axs)

  for i, ax in enumerate(axs):
    ax.matshow(atten[layer][0][i], cmap=plt.cm.Blues)
    ax.set_xticks(range(max_len))
    ax.set_xticklabels(tokens[:max_len], rotation=60)
    ax.set_yticks(range(max_len))
    ax.set_yticklabels(tokens[:max_len])

  plt.tight_layout()
  plt.show()


def show_attention_per_head(model_output, input_ids, head, figsize=(30,50), tokenizer = None):
  max_len = min((input_ids == 102).nonzero(as_tuple=True)[1].item(), 90) +1
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  atten_mtx = torch.stack(model_output[2]).detach().cpu()
  atten = resize(atten_mtx, max_len)
  
  fig, axs = plt.subplots(3,2, figsize = figsize)
  axs = np.ravel(axs)

  for i, ax in enumerate(axs):
    ax.matshow(atten[i][0][head], cmap=plt.cm.Blues)
    ax.set_xticks(range(max_len))
    ax.set_xticklabels(tokens[:max_len], rotation=60)
    ax.set_yticks(range(max_len))
    ax.set_yticklabels(tokens[:max_len])

  plt.tight_layout()
  plt.show()


def show_attention_weights(model_output, input_ids, figsize=(30,30), tokenizer = None):
  max_len = min((input_ids == 102).nonzero(as_tuple=True)[1].item(), 90) +1
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  atten_mtx = torch.stack(model_output[2]).detach().cpu()
  atten = resize(atten_mtx, max_len) * 10000
  
  fig, axs = plt.subplots(6,12, figsize = figsize)

  for i in range(6):
      for j in range(12):
        axs[i][j].matshow(atten[i][0][j], cmap=plt.cm.coolwarm) 
        # ax[i][j].set_xticks(range(max_len))
        # ax[i][j].set_xticklabels(tokens[:max_len], rotation=60)
        # ax[i][j].set_yticks(range(max_len))
        # ax[i][j].set_yticklabels(tokens[:max_len])

  plt.tight_layout()
  plt.show()


def resize(att_mat, max_length=None):
  """Normalize attention matrices and reshape as necessary."""
  att_res = []
  for i, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      att = np.expand_dims(att, axis=0)
    if max_length is not None:
      # Sum across different attention values for each token.
      att = att[:, :, :max_length, :max_length]
      row_sums =  np.sum(att.numpy(), axis=2)
      # row_sums =  att.sum(axis=2)
      # Normalize
      att /= row_sums[:, :, np.newaxis]
    att_res.append(att)
  att_res = np.stack(att_res)
  return att_res

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

def get_most_attented_ds(input_iterator, model, triggers):
  most_attented_tokens = []
  for input in input_iterator:
    model_output, input_ids = utils_model.get_model_output(model, input, triggers, output = 'full', device = 'cuda', eval=True, return_input = True)

    attention_out = model_output[2]
    tokens, ids = get_most_attented(attention_out, input_ids, n_words=5)
    most_attented_tokens.append(tokens)
    
  most_attented_tokens = np.array(most_attented_tokens).flatten()
  tokens_unique, counts = np.unique(most_attented_tokens, return_counts=True)
  sorted_counts = (-counts).argsort()

  return tokens_unique[sorted_counts], counts[sorted_counts]
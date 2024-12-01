from model.types import Input, Output, Features, Labels
from typing import Any, List, Dict
import torch

def featurize(inputs: List[Input], config) -> Features:  
  past_values_list: List[List[List[float]]] = []
  future_values_list: List[List[List[float]]] = []
  static_real_features: List[List[float]] = []
  
  for input in inputs:
    input_cycle_values: List[List[float]] = []
    
    for cycle in input.past_time_cycles:
      
      cycle_features: List[float] = [
        cycle.cycle_duration, 
        cycle.menstruations_duration, 
        cycle.ovulation_day,
        cycle.menstruations_duration,
        cycle.fertility_duration,
        cycle.unusual_bleeding,
      ]

      input_cycle_values.append(cycle_features)
      
    past_values_list.append(input_cycle_values[:config.sequence_length])
    future_values_list.append(input_cycle_values[-config.prediction_length:])
    
    static_real_features.append([
       input.personnal_data.age, 
       input.personnal_data.stress, 
       input.personnal_data.body_temperature
    ])

  past_values: torch.FloatTensor = torch.FloatTensor(past_values_list)
  future_values: torch.FloatTensor = torch.FloatTensor(future_values_list)

  batch_size, sequence_length, input_size = past_values.shape
  batch_size, prediction_length, input_size = future_values.shape

  features = Features(
    past_values = past_values,
    future_values = future_values,
    past_time_features = rangeTensor(
      (batch_size, sequence_length, config.num_time_features)
    ),
    future_time_features = torch.FloatTensor(
      rangeTensor((batch_size, prediction_length, config.num_time_features)) + sequence_length
    ),
    past_observed_mask = torch.BoolTensor(past_values != 0)
  )
  
  return features


def rangeTensor(shape, value=None): 
  return torch.FloatTensor([
    [
      [i for _ in range(shape[2])] if value is None else value for i in range(shape[1])
    ] for _ in range(shape[0])
  ])


def boolTensor(shape, value=1):
  return torch.BoolTensor([
    [
      [value for x in range(shape[2])] for i in range(shape[1])
    ] for _ in range(shape[0])
  ])
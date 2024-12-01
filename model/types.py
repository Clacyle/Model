from typing import TypedDict, List
import torch

class Cycle:
  def __init__(
    self,
    cycle_duration: float,
    menstruations_duration: float,
    ovulation_distribution: List[float]
  ):
    self.cycle_duration = cycle_duration
    self.menstruations_duration = menstruations_duration
    self.ovulation_distribution = ovulation_distribution
    #self.MeanCycleLength = MeanCycleLength 

  
class Human:
  def __init__(self, 
     age: float,
     body_temperature: float,
     stress: float
  ):
    self.age = age
    self.body_temperature = body_temperature
    self.stress = stress


class Input:
  def __init__(self, 
     past_time_cycles: List[Cycle],
     personnal_data: Human
  ):
    self.past_time_cycles = past_time_cycles
    self.personnal_data = personnal_data


class Output(TypedDict):
  pass


# model inputs
class Features:
  def __init__(self, 
    past_values: torch.FloatTensor,
    past_time_features: torch.FloatTensor,
    future_values: torch.FloatTensor,
    future_time_features: torch.FloatTensor,
    past_observed_mask: torch.BoolTensor,
    # static_real_features: torch.FloatTensor,
  ):
    self.past_values = past_values
    self.past_time_features = past_time_features
    self.future_values = future_values
    self.future_time_features = future_time_features
    self.past_observed_mask = past_observed_mask
    # self.static_real_features = static_real_features

  def __str__(self):
    res = "past_values:" + str(self.past_values.shape) + "\n"
    res += "past_time_features:" + str(self.past_time_features.shape) + "\n"
    res += "future_values:" + str(self.future_values.shape) + "\n"
    res += "future_time_features:" + str(self.future_time_features.shape) + "\n"
    res += "\npast_observed_mask:" + str(self.past_observed_mask.shape) + "\n"
    # res += "static_real_features:" + str(self.static_real_features.shape) + "\n"
    
    return res

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, key):
    if key == "past_values":
      return self.past_values
    elif key == "future_values":
      return self.future_values
    elif key == "past_time_features":
      return self.past_time_features
    elif key == "future_time_features":
      return self.future_time_features
    elif key == "past_observed_mask":
      return self.past_observed_mask
    elif key == "static_real_features":
      return self.static_real_features
    else:
      raise KeyError(f"Key {key} not found in Features.")

# model outputs
class Labels:
  pass
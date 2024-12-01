import torch

def training(model, dataset):
  
  for batch in dataset:
    outputs = model(
      past_values = batch["past_values"],
      past_time_features = batch["past_time_features"],
      future_values = batch["future_values"],
      future_time_features = batch["future_time_features"],
      past_observed_mask = batch["past_observed_mask"],
      # static_real_features = batch["static_real_features"],
    )

    loss = outputs.loss
    loss.backward()

  return model, outputs
from model.inference import inference
from typing import List
from model.types import Features
import torch

def benchmark(model, dataset: List[Features]):
  batch_index = 0
  for prediction in inference(model, dataset):
    batch = dataset[batch_index]

    batch_size, sequence_length, input_size = batch.past_values.shape
    batch_size, prediction_length, input_size = batch.future_values.shape
    
    print(f"⏳ Passed {sequence_length} cycles to model, in order to predict {prediction_length} future cycles.\n\t📊 For person n°{batch_index + 1}, prediction => {evaluate(prediction, batch)}")
    
    batch_index += 1


def evaluate(prediction, batch) -> str:
  future_values = batch["future_values"]
  predicted_values = prediction.sequences.mean(dim=1)

  future_values_flat = future_values.view(-1, future_values.shape[-1])
  predicted_values_flat = predicted_values.view(-1, predicted_values.shape[-1])

  diff = (future_values_flat - predicted_values_flat)

  mape = None
  mae = torch.mean(torch.abs(diff)).item()
  rmse = torch.sqrt(torch.mean(diff ** 2)).item()
  if not torch.any(future_values_flat == 0):
    mape = torch.mean(torch.abs(diff / future_values_flat)).item() * 100

  return f"\n\t\tMAE: {mae}\n\t\tRMSE: {rmse}" + (f"\n\t\tMAPE: {mape}" if mape else "")


from model.types import Input, Features
from model.featurize import featurize
from typing import List

def slice(inputs: List[Input], config) -> List[List[Features]]:
  split1 = config.training_data_ratio
  split2 = split1 + config.validation_data_ratio
  
  training = inputs[:int(len(inputs) * split1)]
  validation = inputs[int(len(inputs) * split1):int(len(inputs) * split2)]
  testing = inputs[int(len(inputs) * split2):]

  training_batches = split(training, config.batch_size)
  validation_batches = split(validation, config.batch_size)
  testing_batches = split(testing, config.batch_size)

  def featurizer(batches: List[List[Input]]) -> List[Features]:
    results = []
    for batch in batches:
      results.append(featurize(batch, config))
    return results

  return list(map(featurizer, [
    training_batches,
    validation_batches,
    testing_batches
  ]))


def split(inputs: List[Input], n, lastIncomplete=False) -> List[List[Input]]:
  batches = []
  batch = []
  for i in range(len(inputs)):
    el = inputs[i]
    if i % n == 0:
      batch.append(el)
      batches.append(batch)
      part = []
    part.append(el)

  if lastIncomplete:
    batches.append(batch)
    
  return batches
from model.types import Input, Cycle, Human, Features
from model.slice import slice
from typing import List

def batch(dataset, config) -> List[Features]:
  inputs: List[Input] = []
  sequence_length = config.context_length + max(config.lags_sequence)

  clientIDs = {}

  for record in dataset:
    if record["id"] not in clientIDs:
      clientIDs[record["id"]] = True
  
  for clientID in clientIDs.keys():
    clientRecords = dataset.filter(lambda record: record["id"] == clientID)
    
    if len(clientRecords) <= sequence_length:
      continue
    
    clientCycles = []
    clientData = None
    
    for record in clientRecords:
      clientCycles.append(Cycle(
         cycle_duration = float(record["cycle_duration"]),
         menstruations_duration = float(record["menstruations_duration"]),
         ovulation_distribution = []
      ))

      clientData = Human(
         age = float(record["age"]),
         body_temperature = 37.0,
         stress = 0.0
      )
    
    inputs.append(Input(
      past_time_cycles = clientCycles[-sequence_length:],
      personnal_data = clientData
    ))

  return slice(inputs, config)
from model.types import Input, Cycle, Human, Features
from model.slice import slice
from typing import List

def batch(dataset, config) -> List[List[Features]]:
  inputs: List[Input] = []
  sequence_length = config.sequence_length

  clientIDs = {}
  
  for record in dataset:
    if record["id"] not in clientIDs:
      clientIDs[record["id"]] = True
  
  for clientID in clientIDs.keys():
    clientRecords = dataset.filter(lambda record: record["id"] == clientID)

    if len(clientRecords) < sequence_length:
      continue
  
    clientCycles = []
    clientData = None
    
    for record in clientRecords:
      clientCycles.append(Cycle(
        cycle_duration = getprop(record, "cycle_duration"),
        menstruations_duration = getprop(record, "menstruations_duration"),
        ovulation_day = getprop(record, "ovulation_day"),
        unusual_bleeding = getprop(record, "unusual_bleeding"),
        fertility_duration = getprop(record, "fertility_duration"),
        menstruations_score = getprop(record, "menstruations_score"),
      ))

      clientData = Human(
         age = getprop(record, "age"),
         body_temperature = getprop(record, "body_temperature"),
         stress = getprop(record, "stress"),
      )

    for i in range(0, len(clientCycles), sequence_length):
      cyclesPart = clientCycles[i:i+sequence_length]
      
      if len(cyclesPart) < sequence_length:
        break
        
      inputs.append(Input(
        past_time_cycles = cyclesPart,
        personnal_data = clientData
      ))

  return slice(inputs, config)

def getprop(record, prop):
  if prop in record and record[prop] is not None:
      return float(record[prop])
  return 0.0
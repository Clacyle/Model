from datasets import load_dataset

dataset = load_dataset("efelia-hackathon-team3/Menstrual-Cycle-Data")

renameCols = {
  "LengthofCycle": "cycle_duration",
  "Age": "age",
  "PhasesBleeding": "menstruations_duration",
  "ClientID": "id"
}

dataset = dataset["train"].rename_columns(renameCols)

uselessCols = [col for col in dataset.column_names if col not in [
  "cycle_duration", 
  "menstruations_duration",
  "age",
  "id",
  
  "MeanCycleLength", 
  "EstimatedDayofOvulation", 
  "TotalFertilityFormula",
]]

dataset = dataset.remove_columns(uselessCols)

def replaceBlanks(el):
  if el['menstruations_duration'] == " ":
    el['menstruations_duration'] = '0'

  return el

def replaceBlanks1(el):
  if el['age'] == " ":
    el['age'] = '0'
  return el
  
dataset = dataset.map(replaceBlanks)
dataset = dataset.map(replaceBlanks1)
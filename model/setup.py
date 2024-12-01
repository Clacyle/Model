from transformers import TimeSeriesTransformerConfig
from transformers import TimeSeriesTransformerForPrediction
import json

def setup(configPath: str) -> TimeSeriesTransformerForPrediction:
  
  with open(configPath, "r") as f:
    jsonConfig = json.load(f)
    
  config = TimeSeriesTransformerConfig(**jsonConfig)
  model = TimeSeriesTransformerForPrediction(config)

  return model
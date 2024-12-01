from transformers import TimeSeriesTransformerConfig
from transformers import TimeSeriesTransformerForPrediction

from model.inference import inference
from model.training import training
from model.batch import batch
from model.validate import validate
from model.sampling import distribution

from dataset import dataset
import json

# import config from JSON and build model
jsonConfig = json.load(open("modelconfig.json", "r"))
config = TimeSeriesTransformerConfig(**jsonConfig)
model = TimeSeriesTransformerForPrediction(config)

# build batches from dataset
trainingSet, validationSet, testingSet = batch(dataset, model.config)

# run simple tests about batch features
validate(trainingSet, model.config)

# train the model on the training data
model, outputs = training(model, trainingSet)

# predict new data from the model
for prediction in inference(model, validationSet):
  mean_prediction = prediction.sequences.mean(dim=1)
  print(mean_prediction)
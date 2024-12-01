from model.inference import inference
from model.training import training
from model.batch import batch
from model.validate import validate
from model.sampling import distribution
from model.setup import setup
from model.evaluate import benchmark

from dataset import dataset

# import config from JSON and build model
model = setup("modelconfig.json")

# build batches from dataset
trainingSet, validationSet, testingSet = batch(dataset, model.config)

# run simple tests about batch features
validate(trainingSet, model.config)

# train the model on the training data
model, outputs = training(model, trainingSet)

benchmark(model, validationSet)
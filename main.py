from model.inference import inference
from model.training import training
from model.batch import batch
from model.validate import validate
from model.sampling import distribution
from model.setup import setup
from model.evaluate import benchmark

from dataset import dataset as dataset1
from dataset2 import dataset as dataset2

# import config from JSON and build model
model = setup("modelconfig.json")

# build batches from dataset
trainingSet, validationSet, testingSet = batch(dataset1, model.config)
validate(trainingSet, model.config)
model, outputs = training(model, trainingSet)

trainingSet, validationSet, testingSet = batch(dataset2, model.config)
validate(trainingSet, model.config)
model, outputs = training(model, trainingSet)

benchmark(model, testingSet)
def inference(model, dataset):
  
  for batch in dataset:
    yield model.generate(
      past_values=batch["past_values"],
      past_time_features=batch["past_time_features"],
      past_observed_mask=batch["past_observed_mask"],
      future_time_features=batch["future_time_features"],
    )


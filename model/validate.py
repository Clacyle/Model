def validate(dataset, config):
  for batch in dataset:
    assert batch.past_values.shape[1] == config.sequence_length
    assert batch.future_values.shape[1] == config.prediction_length
    assert batch.future_time_features.shape[-1] == config.num_time_features
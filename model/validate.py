def validate(dataset, config):

  assert config.sequence_length == config.context_length + max(config.lags_sequence)
  
  for batch in dataset:
    assert batch.past_values.shape[1] == config.sequence_length
    assert batch.future_values.shape[1] == config.prediction_length
    assert batch.future_time_features.shape[-1] == config.num_time_features

  print("✅ Well-formed dataset and hyper-parameters")
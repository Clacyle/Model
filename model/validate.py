def validate(dataset, config):

  err = "❌ Inconsistency of dataset shapes or hyper-parameters"
  assert config.sequence_length == config.context_length + max(config.lags_sequence), err
  
  for batch in dataset:
    assert batch.past_values.shape[1] == config.sequence_length, err
    assert batch.future_values.shape[1] == config.prediction_length, err
    assert batch.future_time_features.shape[-1] == config.num_time_features, err

    if config.input_size != 1:
      assert batch.past_values.shape[-1] == config.input_size, err
      assert batch.future_values.shape[-1] == config.input_size, err

  print("✅ Well-formed dataset and hyper-parameters")
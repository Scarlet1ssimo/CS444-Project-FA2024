class TrainConfig:
    max_depth = 26                          # God's Number
    batch_size_per_depth = 1000
    num_steps = 10000
    learning_rate = 1e-3
    INTERVAL_PLOT, INTERVAL_SAVE = 100, 1000
    # Set this to True if you want to train the model faster
    ENABLE_FP16 = False


class SearchConfig:
    # This controls the trade-off between time and optimality
    beam_width = 2**11
    max_depth = TrainConfig.max_depth * 2   # Any number above God's Number will do
    # Set this to True if you want to solve faster
    ENABLE_FP16 = False

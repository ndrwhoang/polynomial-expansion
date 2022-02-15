# How to Run
In the `main.py` file there are 2 functions, use `run_train()` to start the training process and `run_test()` to run inference. The file paths and checkpoint locations are according to `config.cfg`. `test_dataset` in `data_path` should lead to the test file and `model_ckpt` is the trained model.

# Modeling
I went with the 'tried and true' transformer encoder-decoder stack and push the number of parameters just below the 5M limit (add as many heads and layers as possible, i.e. making the model as deep as possible, then make the intermediary as wide as possible).

# Data split 
Train set: 640K samples  
Valid set: 160K samples
Test set: 200k samples  

# Training
Batch size: 64  
Lr: 1e-3
Scheduler: None  
Save checkpoint at lowest validation loss   

# Result
Test accuracy (perfect match): 0.849

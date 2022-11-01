## Usage of T5Wrapper

```
Note: This library is adapted from SimpleT5 library (https://github.com/Shivanandroy/simpleT5) 
to expose some parameters for T5. The main changes are:
- Allowing predictions to be done in batches instead of single instances
- returning the logit scores along with the prediction. This is required to do threshold analysis.
- This also adds support to load custom pre-trained flax models
```

### Usage

```python
from T5Wrapper import T5Wrapper
model = T5Wrapper()

## Load pre-trained flax model
model.from_pretrained(model_type="pre_trained", model_name="path_to_dir")

## load existing checkpoint
model.from_pretrained(model_type="t5", model_name="t5-small")

model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=256, 
            target_max_token_len=4, 
            batch_size=8, max_epochs=10, use_gpu=True, early_stopping_patience_epochs=3, outputdir="pre_trained_binary")

```
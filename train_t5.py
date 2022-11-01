import pandas as pd
from sklearn.model_selection import train_test_split

import os

print("CUDA VISISBLE DEVICES = {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

#from simplet5 import SimpleT5
from T5Wrapper import T5Wrapper

train_df = pd.read_json('./binary_train.json')
test_df = pd.read_json('./binary_test.json')

# train_df = train_df.drop(['domain'],axis=1)
# test_df = test_df.drop(['domain'],axis=1)

print("Length of the dataframe: {}".format(len(train_df)))

# train_df, test_df = train_test_split(df, test_size=0.2)
print(train_df.shape, test_df.shape)

model = T5Wrapper()
#model.from_pretrained(model_type="t5", model_name="t5-large")
model.from_pretrained(model_type="pre_trained", model_name="/home/jingjie/Github/Pretrain/t5-base_policy")

print("Starting model training")

model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=256, 
            target_max_token_len=4, 
            batch_size=8, max_epochs=10, use_gpu=True, early_stopping_patience_epochs=3, outputdir="pre_trained_binary")

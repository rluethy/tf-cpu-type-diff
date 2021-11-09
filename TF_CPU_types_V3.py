# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # Demonstrate differences when using Tensorflow on different CPU types

# + slideshow={"slide_type": "subslide"}
import os
import subprocess
import numpy as np
import pandas as pd
from IPython.display import display
# %matplotlib inline
import matplotlib.pyplot as plt
import random as rn
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Softmax

pd.set_option("display.float_format", '{:.3f}'.format)
# +
def getCPUtype():
    cpuinfo = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode()
    for l in cpuinfo.split("\n"):
        if l.find("vendor_id") >= 0:
            if l.find("GenuineIntel") >= 0:
                return "Intel"
            else:
                return "AMD"

def gen_iterator_rand(mini_batch_size, dataset, labels):
    while True:
        rand_idx = np.random.choice(range(dataset.shape[0]), mini_batch_size, False)
        yield dataset[rand_idx, :], labels[rand_idx].astype('int32')


# -

# ## Load data

# +
CPUtype = getCPUtype()
print("CPU type:", CPUtype)

train_df = pd.read_csv("training_data.csv", index_col=0)
val_df = pd.read_csv("validation_data.csv", index_col=0)
print("Data shapes:", train_df.shape, val_df.shape)
display(train_df.head())
# -

# ## Make model 

# +
tf_seed = 371
batch_size = 128
iteration = 50
hidden_layers = 3
learning_rate = 0.1
hidden_units = 20

# set the various random seeds
np.random.seed(888)
rn.seed(12345)

tf.random.set_seed(tf_seed)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

initialization = tf.keras.initializers.GlorotUniform(seed=2)

dense_input = tf.keras.Input(shape=(29,), dtype=tf.float32)

a = Dense(
    units=hidden_units,
    activation="relu",
    kernel_initializer=initialization
)(dense_input)

count = 1
for i in range(hidden_layers - 1):
    a = Dense(
        units=hidden_units,
        activation="relu",
        kernel_initializer=initialization
    )(a)
    a = BatchNormalization()(a)

# final layer 
predictions = Dense(
    units=2,
    activation="softmax",
    kernel_initializer=initialization
)(a)

model = Model(inputs=dense_input, outputs=predictions)

adam_opt = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    name="Adam"
)
model.compile(optimizer=adam_opt, loss='sparse_categorical_crossentropy', run_eagerly=False)
model.summary()

# -

# ### Fit model

start_iter=0
np.random.seed(tf_seed)
logfile = open(f"batches_{CPUtype}.log","w")
gen_batch = gen_iterator_rand(batch_size,train_df.drop(["truth"], axis=1).values, 
                                   labels=train_df["truth"].values)
for i in range(start_iter, iteration):
    batch_x, batch_y = next(gen_batch)
    for i,b in enumerate(batch_x):
        logfile.write(repr(i) +" "+ repr(b) +"\n")
    logfile.write("\n")
    #print(batch_x.shape, batch_y.shape)
    model.train_on_batch(batch_x, batch_y)
logfile.close()
val_df["prob_1"] = model.predict(val_df.drop(["truth"], axis=1).values)[:,1]
#display(val_df)
val_df.to_csv(f"val_preds_{CPUtype}.csv")

# ! diff -s batches_AMD.log batches_Intel.log 

# ### Plot Intel vs. AMD

if os.path.exists("val_preds_AMD.csv") and os.path.exists("val_preds_Intel.csv"):
    amd = pd.read_csv("val_preds_AMD.csv")
    intel = pd.read_csv("val_preds_Intel.csv")
    print((intel["prob_1"] - amd["prob_1"]).abs().sort_values(ascending=False).head(10))
    fg = plt.figure(figsize=(8,8))
    plt.scatter(intel["prob_1"], amd["prob_1"])
    plt.title("Prob 1")
    plt.xlabel("Intel")
    plt.ylabel("AMD")
    plt.show()



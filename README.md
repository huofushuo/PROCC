## Setup 

1. Clone the repo 

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
    conda env create --file environment.yml
    conda activate procc
```
3. Go to the cloned repo and open a terminal. Download the datasets and embeddings, specifying the desired path (e.g. `DATA_ROOT` in the example):
```
    bash ./utils/download_data.sh DATA_ROOT
    mkdir logs
```

## Training
**Open World.** To train a model, the command is simply:
```
    python train.py --config CONFIG_FILE 
```
where `CONFIG_FILE` is the path to the configuration file of the model. 
The folder `configs` contains configuration files for all methods, i.e. CGE in `configs/cge`, CompCos in `configs/compcos`, and the other methods in `configs/baselines`.  

To run ProCC on MIT-States, the command is just:
```
    python train.py --config configs/procc/mit.yml --open_world --fast
```
On UT-Zappos, the command is:
```
    python train.py --config configs/procc/utzappos.yml --open_world --fast
```

**Partial Label Setting** To train ProCC (in the partial label setting) on MIT-States, run:
```
    python train.py --config configs/procc/partial/mit.yml --partial --fast
```

**Note:** To create a new config, all the available arguments are indicated in `params.py`. 

## Test


**Open World.** To test a model in the open world setting, run:
```
    python test.py --logpath LOG_DIR --open_world --fast
```

**Partial Label Setting** To test a ProCC model on the partial label setting, a similar command can be used:
```
    python test.py --logpath LOG_DIR --fast --partial
```

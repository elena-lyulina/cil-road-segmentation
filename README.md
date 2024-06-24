# ETHZ Spring 2024 CIL Road Segmentation Project

### Deadline:
✨July 31st 23:59✨

### Environment Setup
To replicate our experiments, follow these steps to create and activate a virtual environment:

#### For Windows:
```bash
python -m venv ./cil-rs
cil-rs\Scripts\activate
```

#### For MacOS and Linux:
```bash
python -m venv ./cil-rs
source cil-rs/bin/activate
```

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

Note: We run our experiments on Python version 3.11.5. We recommend users use the correct Python executable to initiate the virtual environment.

### Directory Structure
```
CIL-ROAD-SEGMENTATION-2024
└───cil-rs
    |───data                        // various datasets used for this task
    |   |───cil
    |   |   |───test
    |   |   |   └───images
    |   |   └───training
    |   |       |───groundtruth
    |   |       └───images
    |   └───data-massachusetts
    |       |───test 
    |       |───test_labels
    |       |───train
    |       |───train_labels
    |       └───val
    |───docs
    |───notebook
    |───out                         // models for submission
    └───src                         // source code
        |───data                    // datasets, dataloaders, preprocessing
        |   |───dataset_1.py
        |   |───datasethandler.py   // a helper class to handle data splitting for train/test
        |   └───utils.py            // any common functions used for datasets
        |
        |───experiments             // all experiments we've done
        |   |───exp_1
        |   |   |───results         // results of the experiment: trained models, etc
        |   |   └───main.py         // running the experiment
        |   |───config.py           // common code to configure an experiment
        |   |───registry.py         // means to register datasets, models, any other params for running an experiment
        |   └───utils.py            // any common code used for running the experiments
        |
        |───models                  // implementation of the models
        |   |───model_1
        |   └───utils.py            // any common code used for implementing the models
        |
        |───submission              // sumission code
        |
        |───train                   // training the models
        |   |───metrics.py          // metrics calculated during training
        |   |───train.py            // the main implementation of the training
        |   └───utils.py            // any common code needed for training
        |
        └───constants.py            // various constants used in the projects
```
### Configs and registry
We're using ```.json``` configs to control every run: models, datasets, training parameters.

For example, the following config would train a ```small_unet``` model on the ```cil``` dataset: 
```
{
    'model': {
        'name': 'small_unet',                         // a name the model is registered under
        'params': {                                   // all the necessary params to pass into __init__ method of the model implementation,
            'chs': (3, 64, 128, 256, 512, 1024)       // may vary for different models
        }
    },
    'dataset': {
        'name': 'cil',                                // a name the dataset is registered under
        'params': {                                   // all the necessary params to pass to this dataset's datahandler,
            'batch_size': 4,                          // may vary for different datasets
            'shuffle': True,
            'resize_to': (384, 384)
        }
    },
    'train': {                                        // parameters for training, find the default config in src.train.utils.py
        'n_epochs': 1,
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.0005
            }
        },
        'loss': {
            'name': 'BCELoss',
            'params': { }
        },
        'clip_grad': None
    }
}
```
For the models and datasets to be found and loaded by a config, they need to be registered first. 
This is done via ```src.experiments.registry.Registry```.
Just annotate a new implementation of a model or a dataset specifying its name, and it will become visible for any config you run. 
More details about it you'll find in the corresponding sections. 

### Adding a new dataset
All data is stored under ```/data``` folder. Please, create a subfolder and add your new dataset there.

For every dataset you need to implement a ```torch.utils.data.Dataset``` class to use during training.
Additionally, since we may handle data splitting differently for every dataset, there is a ```src.data.datahandler.DataHandler```  class
that needs to be implemented for every new dataset.
It's responsible for data splitting and anything else needed to load the correct data from a config file.

Once these two classes are implemented, annotate them with ```@DATASET_REGISTRY.register("<unique_name>")``` and ```@DATAHANDLER_REGISTRY.register("<unique_name>")```. 
Use this name in a config to train a model on this specific dataset. 

### Implementing a new model

Add a new model implementation in a subfolder under ```src.models```.
Don't forget to register it in a similar way by adding ```@MODEL_REGISTRY.register("<unique_name>")``` and pray that it works.

### Training a model

We're maintaining a separate folder for running all experiments: ```src.experiments```. 
To run something new, create a subfolder with a meaningful name, and you'll free to do anything you want there without messing with any other people's work. 

A simple way to run something would be creating a ```main.py``` file there, a config dictionary, and calling ```src.experiments.config.run_config``` with the created config. 
```src.experiments.config.run_config``` with the created dict.

You will also need to pass a path to save the results to and the name of the experiment.
Please, save everything in the experiment folder to not clutter the rest of the repository (unless you really need to). 
To save you some bothers, there is a function ```src.experiments.utils.get_save_path_and_experiment_name``` you can call, passing ```__file__``` argument. 
This way it will return a path to a ```results``` subfolder and the name of the current experiment subfolder (but it only works with the default file structure).  

You also need to name the current run since you will probably have more than one for any experiment, e.g. if you're trying out different parameters.  
For that there is also a function ```src.experiments.utils.get_run_name``` which will return you a combination of the names of the current model/dataset and any suffix you want to add.

Finally, you will need to create a config: a dictionary with all settings for the current run.
Of course, you can do it manually, but there is also a helper function for that.
Just call ```src.experiments.config.generate_config``` with the registered names of the model and the dataset you want to use.
It will print a dictionary with all the necessary params you need to fill in to later be passed into the corresponding model/datahandler class (taken from their ```init``` method). 

Have fun!

##### Troubleshooting
It might be that the config cannot indicate the names of the model / dataset.
Double check that you have added the required annotations for the model, the dataset, and the datahandler.
If that doesn't help, try explicitly import these classes in the ```src/experiments/config.py``` file, since the registry only works with the classes it can see.
However, this should work by default as long as you added your implementations under ```src/models``` and ```src/data``` folders.  

### Testing a model
...to be implemented...
### Submitting a model
... to be implemented...

### Connecting Weights & Biases
It's always good to log all the experiments, and for that we're using [Weights & Biases](https://wandb.ai/site) framework.  
After being added to our project organization, you can start using it for this project. 

Make sure you have ```wandb``` installed or run
```bash
pip install wandb
```

Log in locally to your account and paste your API key when prompted:
```bash
wandb login
```

After that, log your runs by passing ```log_wandb=True``` to ```src.experiments.config.run_config```.


### Links: 
1. [Video recording](https://video.ethz.ch/lectures/d-infk/2024/spring/263-0008-00L/fe8cb982-d061-4350-8c3e-26b0cdb43119.html) of the preparatory session about this project
2. [Kaggle link](https://www.kaggle.com/t/0fe22c50cf504e64b2decda075f71c87)
3. [Massachusetts Road Dataset download](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset/data?select=tiff)

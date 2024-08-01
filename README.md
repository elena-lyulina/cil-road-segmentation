# ETHZ Spring 2024 CIL Road Segmentation Project

This repository contains the implementation of a road segmentation data-specific pipeline, developed as part of a Kaggle competition by team KidNamedFinger.


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

### Dataset
We assembled a large dataset of satelite images based on thorough analysis of given samples to ensure high-quality, relevant training data, split into clusters.
All data used in this study can be downloaded here. 
Jupyter notebook ```notebook/cil_data.ipynb``` contains all data analysis, visualization, filtration, and clustering.

### Trained models
We experiemnted with many different SOTA models and various architectures to predict road masks and restore roads connectivity.
All trained models can be downloaded [here](https://polybox.ethz.ch/index.php/s/j6IySJc5mwSxeoF).

### Results reproduction
To reproduce the results, you can run ```src/submission/evaluate.py``` on dowloaded models and data. 
As both models and data are quite heavy, the results are additionally saved in the ```out``` directory. 

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
    |   └───...
    |
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
        |   |───example
        |   |   |───results         // results of the experiment: trained models, etc
        |   |   |───finetune.py     // finetuning a model
        |   |   |───main.py         // running the experiment
        |   |   └───sweep.py        // running a sweep
        |   |
        |   |───config.py           // common code to configure an experiment
        |   |───registry.py         // means to register datasets, models, any other params for running an experiment
        |   |───sweep_config.py     // sweep configuration 
        |   └───utils.py            // any common code used for running the experiments
        |
        |───models                  // implementation of the models
        |   |───model_1
        |   └───utils.py            // any common code used for implementing the models
        |
        |───submission              // submission code
        |   |───end2end.py          // whole pipeline submission
        |   |───evaluate.py         // evaluating models with configs on various data
        |   |───mask_to_submission.py
        |   |───submission_to_mask.py
        |   └───test.py             // test and create a submission
        |
        |───train                   // training the models
        |   |───loss.py             // loss functions
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
        'params': {                                   // all the necessary params to pass into the __init__ method of the model implementation,
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

Add a new model implementation in a subfolder under ```src/models```.
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

PS. See `src/experiments/example/main.py` for an example. 


##### Troubleshooting
It might be that the config cannot indicate the names of the model / dataset.
Double check that you have added the required annotations for the model, the dataset, and the datahandler.
If that doesn't help, try explicitly import these classes in the ```src/experiments/config.py``` file, since the registry only works with the classes it can see.
However, this should work by default as long as you added your implementations under ```src/models``` and ```src/data``` folders.  

### Training a pretrained model
It is also possible to fine-tune a pretrained model on more data.
For that you need to have the config file of the pretrained model and the model file itself.

First, generate the fine-tuning config by running `src.experiments.config.generate_finetuning_config`,
passing the path to the config file and the name of the dataset to fine-tune it on. 

Then you can run this config the same way as before. 

See `src/experiments/example/finetune.py`. 

### Testing & submitting a model
To submit a model, it needs to be run on test images from the CIL dataset. 
Assuming the model takes whole images as input (possible resized) and returns a prediction for every pixel,
there is a helper function to run the model on test images and create a submission file: `src.submission.test.test_on_full_images`.
Just pass the path to the model's config there and (optionally) the resize dimensions.
Note that if the config file used the CIL dataset, the resize dimensions will be taken from there. 

Check `src/experiments/example/submit.py`.

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

### W & B: Tune Hyperparameters

You can easily [run a sweep](https://docs.wandb.ai/guides/sweeps) with Weights & Biases to try out different hyperparameters for your model. 
Look at `src/experiments/example/sweep.py` for an example, or follow these steps:


**Step 1.** Prepare a config by first generating a usual one and then changing the parameters you want to tune
by adding a prefix `"SWEEP_"` to their name and providing a list of values / distribution to try out.

For example, 
```
"lr": 0.01    =>    "SWEEP_lr": { "values" : [0.001, 0.05, 0.01] }
                         
              OR
                
"lr": 0.01    =>    "SWEEP_lr": { "distribution": "uniform", "min": 0.001, "max": 0.1 }
```

**Step 2.** Create a sweep config by running `src.src.experiments.sweep_config.get_sweep_config`, and use it 
to initialize a sweep by running `src.experiments.sweep_config.init_sweep`.

Use the created sweep id for the next step.

**Step 3.** Now, you can run an agent for the created sweep by passing the received sweep id into `src.experiments.sweep_config.run_sweep_agent`,
specifying the number of runs to try. 


### End-to-End Submission

To create a submission using the entire pipeline, you can use the `src.submission.end2end` module. 
Similar to training a model, create an `end2end.py` file in the `src/experiments` directory and 
call the `src.submission.end2end.run_end2end` function with the path to the model's config file.
Refer to `src/experiments/example/end2end.py` for an example.

To use clustering, pass a list of tuples containing the two models for the respective clusters as `config_paths` 
and set the `clusters`-flag to `True`.
By using a simple list of model paths, clustering will not be used and all data will be evaluated on all models.
Choose one of the available voting methods by setting `voter` to either `hard-pixel`, `soft-pixel`, `hard-patch` or `soft-patch`.
Setting the `with_mae`-flag requires you to specify a path to the MAE model in the `mae_path`-parameter
and will use the MAE on top of the voting for post-processing. Otherwise, the final submission will be created using the voting results.

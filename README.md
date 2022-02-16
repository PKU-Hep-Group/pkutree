# pkutree

A demo to use [coffea](https://github.com/CoffeaTeam/coffea) to do analysis.

### Contents
- `config`: Subfolders with common configurations
   - `dataset_cfg.yaml`: define:
     - the channel (e.g., ssww)
     - the NanoAOD version (e.g., nanov7)
     - the sample type (e.g., data, mc)
     - the related datasets configurations (e.g., mc_2018_ssww_nanov7.yaml)
   - `logging_cfg.yaml`: common settings for `logging` module
   - `path_cfg.yaml`: the input and output folders for results

- `data`: External inputs used in the analysis: scale factors, corrections ...

- `datasets`: The sepecific configurations for sample dataset path from [DAS](https://cmsweb.cern.ch/das/), better to keep consistent to settings [checkoutNano](https://github.com/PKU-Hep-Group/checkoutNano/tree/main/datasets)
  
- `module`: Place to define the object selection, event selection modules for different analyses.

- `rundoc`: The folder to keep some output information while running the scripts automatically.

- `utils`: The place for some helper functions.

- `setup.py`: File for installing the package in the `--editable` way.

### Set up the environment 

First, download and install conda:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
Next, run `unset PYTHONPATH` to avoid conflicts. Then run the following commands to set up the conda environment:    
```
conda env create -f environment.yml
conda activate xcu
```

### How to start

- This directory is set up to be installed as a python package. To install, activate your conda environment, then run this command from the top level `pkutree` directory:
```
pip install -e .
```
The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall pkutree`.

- Next, edit the `config/dataset_cfg.yaml` after put your own yaml configuration in datasets folder. And change the `config/path_cfg.yaml` with your own areas.
  
- Put your own module like `moudle/hmm/sel_module`, and set the `utils/step_helper.py` to define the step for your analysis
  
- Lastly, run the script `go_run.py`


```
# run on SingleMuon_Run2018A dataset
python3 go_run.py -c hmm -y 2018 --step obj_sel -s SingleMuon_Run2018A -v nanov7 --chunksize 500000 -d 
# run on MC WZZ dataset
python3 go_run.py -c hmm -y 2018 --step obj_sel -s WZZ -v nanov7 --chunksize 500000
# run all MC
python3 go_run.py -c hmm -y 2018 --step obj_sel -v nanov7 --chunksize 500000 -a

# make plots
python go_plot.py hist -y 2018 -c hmm -f parquet --chunksize 100000 --nuisance
# consider uncertainty
python go_plot.py hist -y 2018 -c hmm -f parquet --chunksize 100000 --nuisance --reorganize
```


### Some references:

The code refer to following urls a lot:
> https://github.com/TopEFT/topcoffea
> https://github.com/CoffeaTeam/coffea/tree/master/tests
> https://coffeateam.github.io/coffea/index.html

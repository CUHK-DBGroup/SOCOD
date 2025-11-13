# SO-COD

## Environment
- Ubuntu
- Python (version 3.11)

## Preparation

First, create the necessary folders:

```sh
mkdir ./data
mkdir ./reslut
```

Place the datasets into the folder `[data]`. 


## Install dependencies:

You can install the required dependencies using `pip` or `conda`:

```sh
pip install numpy tqdm
# or
conda install numpy tqdm
```

## Execution

To execute the algorithms on all datasets, simply run the `run_all.sh` script. The results will be saved in the `[result]` folder. 

```sh
./run_all.sh
```

If you want to execute the algorithm on a specific dataset, use the following command:

```sh
python ./run_all_algorithm.py [dataset_name]
```

Replace `[dataset_name]` with the name of the dataset you want to use.

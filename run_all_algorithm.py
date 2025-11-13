from socod import run_socod
from ehcod import run_ehcod
from dicod import run_dicod
from sampling import run_sampling
import sys

if __name__ == "__main__":
    dataset = [sys.argv[1]]
    
    for dataset_name in dataset:
        print("Running tests for dataset: {}".format(dataset_name))
        run_socod(dataset_name, 50)
        run_ehcod(dataset_name, 50)
        run_sampling(dataset_name, 50)
        run_dicod(dataset_name, 50)
        
    print("All tests passed.")
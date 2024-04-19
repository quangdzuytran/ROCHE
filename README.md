# Robust Estimation of Causal Heteroscedastic Noise Models (ROCHE)
This is the implementation of our paper: 
Quang-Duy Tran, Bao Duong, Phuoc Nguyen, and Thin Nguyen. [Robust Estimation of Causal Heteroscedastic Noise Models (ROCHE)](https://epubs.siam.org/doi/10.1137/1.9781611978032.90). In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM), 2024.

## Dependences 
The configuration for the conda environment is available at [```conda.yml```](conda.yml).

## Running Experiments
To run the experiments, use the [```run.py```](run.py) with the following configurations:
- ```--method roche```: using ROCHE as the method
- ```--data [DATASET]```: the benchmark dataset

To see all available configurations for each experiment, run the python file with ```-h``` or ```--help```.

## Acknowledgement
This code is based on the implementation of [Location-Scale Causal Inference (LOCI)](https://github.com/aleximmer/loci)

## Citation
If you find our code helpful, please cite us as:
```
@inproceedings{tran2024robust,
  author = {Tran, Quang-Duy and Duong, Bao and Nguyen, Phuoc and Nguyen, Thin},
  booktitle = {Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
  title = {Robust Estimation of Causal Heteroscedastic Noise Models},
  year = {2024},
  pages = {788--796},
}
```

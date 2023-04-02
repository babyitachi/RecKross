# RecKross: Optimizing Recommendations with k-Cross Kernel Net


## 1. Introduction
The proposed RecKross is a kernel-based recommendation algorithm. The proposed model cosists of 1) 2D Kernel layer 2) k-Cross Kernel layer. Achieving state-of-the-art performance on the Movielens-1M and Movielens-100K datasets, moreover giving highly competitive performance on other datasets, beating almost all the existing models.


## 2. Requirements
* numpy
* pytorch

## 3. Code execturion commands
1. ./run.sh file_dir_path_of(movielens_1m_dataset.dat) config_file(config.json)

## 4. RMSE Results
- ML-1M: 0.8224
- ML-100K: 0.8912
- Douban: 0.7255


## 4. Data References
1. Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. *Acm transactions on interactive intelligent systems (tiis)*, 5(4), 1-19.

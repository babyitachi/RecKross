# Intentions are more than enough: leveraging user-item intent for the Advanced Recommender system, IntentRec


## 1. Introduction
The proposed IntnetRec is a neural recommendation algorithm. The proposed model cosists of 1) Intent layer 2) Kernel layer. Achieving state-of-the-art performance on the Movielens-100K and Douban datasets, beating all the existing models.


## 2. Requirements
* numpy
* pytorch

## 3. Code execturion commands
1. ./run.sh file_dir_path_of(dataset) config_file(config.json)

## 4. RMSE Results
- Douban: 0.7208
- ML-100K: 0.8858
- ML-1M: 0.823


## 4. Data References
1. Harper, F. M., & Konstan, J. A. (2015). The movielens datasets: History and context. *Acm transactions on interactive intelligent systems (tiis)*, 5(4), 1-19.

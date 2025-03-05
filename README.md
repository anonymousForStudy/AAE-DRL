This repository is the authors' implementation of AAE-DRL, an Intrusion Detection System (IDS) that utilizes the capabilities of an Adversarial Autoencoder (AAE), a hybrid Deep Reinforcement Learning (DRL) algorithm: a Twin-delayed Deep Deterministic Policy Gradient (TD3) and Double Deep Q-Network (DDQN) as well as the TabNet classifier to replicate intrusion behavior and predict its classes. 
The development of each component in AAE-DRL is based on the following implementations:
- [AAE repository](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py)
- [DRL repository (TD3)](https://github.com/sfujim/TD3)
- [TabNet repository](https://github.com/sourabhdattawad/TabNet/tree/master)

Additional links:
- [YData Profiler (repository)](https://github.com/ydataai/ydata-profiling)
- [Optuna Scikit-Learn classifiers (py file)](https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_simple.py)
- [Optuna Extreme Gradient Boosting (py file)](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py)
- [MLFlow (documentation)](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)



# Data Processing
We used the UNSW-NB15 dataset by downloading [the CSV files](https://research.unsw.edu.au/projects/unsw-nb15-dataset), the final version of the dataset (joined, cleaned and preprocessed) is available in data/prep.zip .



# A step-by-step guide to implementing AAE-DRL
To install required packages, run:
```bash
pip install -r requirements.txt
```

## 1. AAE training and testing on unaugmented data
1.1. To train the AAE model on the unaugmented dataset, run:
```bash
python AAE_main.py --train --unaug_dataset
```

1.2. To test the AAE model on the unaugmented dataset, run:
```bash
python AAE_main.py --unaug_dataset
```

*Options*
- --batch_size_train: default=32, type=int : set training batch size
- --batch_size_test: default=64, type=int : set validation and testing batch size
- --numEpochs: default=101, type=int : set number of epochs
- --loss_threshold: default=0.5, type=float : when the discriminator loss reaches the threshold, we save the state dictionary
- --n_inter: default=5, type=int : number of interpolations
- --n_samples_per_inter: default=27321, type=int : number of samples to interpolate through
  n_inter * n_samples_per_inter = total samples
- --train: action='store_true' : train and validate model, if not specified then model testing
- --unaug_dataset: action="store_true" : use unaugmented dataset, if not specified then augmented dataset
- --dataset_file: default="ds.csv" : assign name to dataset
- --save_state_dict: default="aae3.pth" : assign name to state dictionary

**Generator loss graph over 101 epochs on unaugmented data (using MLFlow)**
![G_loss_unaug](https://github.com/user-attachments/assets/2caf65aa-375c-4925-8845-3a8ec15e4d57)

**Discriminator loss graph over 101 epochs on unaugmented data (using MLFlow)**
![D_loss_unaug](https://github.com/user-attachments/assets/77cf2765-37b5-418a-91b2-10beda92dfe2)


## 2. Benchmark classification of unaugmented data
2.1. To evaluate the generated data on Gradient Boosting, run:
```bash
python benchmark_clf_main.py --unaug_dataset
```

2.2. To evaluate the generated data on the other benchmark classifiers, set one of the following options to True:
--xgb_clf
--KNN_clf
--rf_clf
Example:
```bash
python benchmark_clf_main.py --rf_clf=True --unaug_dataset
```

On another note, we provided the code for the Optuna trials in the clfs/clf_optim.py file. Accordingly, ou can change the parameters of each classifier in the clfs/benchmark_classification.py file

*Options*
- --features: default="ds_org.csv" : equivalent of --dataset_file ; refer to the dataset generated
- --unaug_dataset : action = "store_true" : use unaugmented dataset, if not specified then augmented dataset
- --xgb_clf: default= False, type=bool : to use Extreme Gradient Boosting
- _-KNN_clf: default= False, type=bool : to use K Nearest Neighbor
- --rf_clf: default= False, type=bool : to use Random Forest

**Classification report of Random Forest tested on unaugmented data**
![RF_unaug](https://github.com/user-attachments/assets/dc253fa4-316f-40be-b7bc-2874a0598c7a)

**Classification report of Extreme Gradient Boosting tested on unaugmented data**
![XGB_unaug](https://github.com/user-attachments/assets/c934836e-adc9-41f5-800d-773c3ea3bf4c)

**Classification report of Gradient Boosting tested on unaugmented data**
![gb_unaug](https://github.com/user-attachments/assets/69db63d9-70c6-418f-8fe4-713f16b0ab5d)

**Classification report of K Nearest Neighbor tested on unaugmented data**
![KNN_unaug](https://github.com/user-attachments/assets/34ddb57b-703a-423f-974f-cf69656c0f69)


## 3. TabNet classifier pre-training 
3.1. To pre-train the classifier, run the original (supervised) dataset that was used to train the AAE:
```bash
python classifier_main.py --train
```

3.2. To test the classifier, run:
```bash
python classifier_main.py
```

*Options*
- --batch_size_train: default=32, type=int : set training batch size
- --batch_size_test: default=64, type=int : set validation and testing batch size
- --numEpochs: default=51, type=int : set number of epochs
- --loss_threshold: default=0.5, type=float : when the classifier loss reaches the threshold, we save the state dictionary
- --train: action='store_true' : train and validate model, if not specified then model testing
- --unaug_dataset: default=True : set to True by default, optionally can set it to False to pre-train augmented dataset
- --save_state_dict: default="clf.pth" : assign name to classifier state dictionary


## 4. DRL training and testing
4.1. To train the DRL algorithm and generate new synthetic data, run:
```bash
python DRL_main.py --train
```

4.2. To test the DRL algorithm, run:
```bash
python DRL_main.py
```

*Options*
- --batch_size_train: default=32, type=int : set training batch size
- --batch_size_test: default=64, type=int : set validation and testing batch size
- --numEpochs: default=100, type=int : run test over n epochs
- --max_timestep: default=4000, type=int : maximum environment runs
- --eval_freq: default=400, type=int : evaluate every n episodes
- --start_timestep: default=50, type= int : stop exploration at n timestep
- --max_ep_steps: default=100, type=int : delayed policy updates frequency
- --train: action='store_true' : train and validate model, if not specified then model testing
- --unaug_dataset: default=True : set to True by default, optionally can set it to False to pre-train augmented dataset
- --rl_dataset: default="rl_ds.csv" : assign name to the generated dataset
- --actor_path: default = "actor.pth" : assign name to actor state dictionary
- --critic_path: default = "critic1.pth" : assign name to critic state dictionary


## 5. TabNet classifier label prediction
To predict labels for the new synthetic dataset, run
```bash
python classifier_main.py --label_gen
```

*Options*
In addition to the options mentioned in STEP 3, we can pass the following:
- --label_gen: action="store_true" : generate labels
- --synth_dataset_path: default="rl_ds.csv" : path to generated dataset
- --labels_file: default = "labels.csv" : assign name to labels generated

  
## 6. AAE training on augmented data
To train the AAE model on the augmented dataset, run:
```bash
python AAE_main.py --train --n_iter=4 --n_samples_per_iter=43313 --dataset_file=<rename_dataset> --save_state_dict=<rename_state_dicts>
```
*Options*
In addition to the options mentioned in STEP 1, we can pass the following:
- --X_ds: default="rl_ds.csv" : refer to the DRL-generated dataset 
- --y_ds: default="labels.csv" : refer to the labels generated
We assign 4 to --n_iter and 43313 to --n_samples_per_iter to match the augmented dataset size. We also assign new values to --dataset_file and --save_state_dict to avoid overwriting files.

**Generator loss graph over 101 epochs on augmented data (using MLFlow)**
![Screenshot from 2025-03-01 07-32-17](https://github.com/user-attachments/assets/67f2d44a-0e0b-44f2-ab27-608a9ed31341)

**Discriminator loss graph over 101 epochs on augmented data (using MLFlow)**
![Screenshot from 2025-03-01 07-31-49](https://github.com/user-attachments/assets/611437d5-b4a3-4239-9bce-6d53dfa57941)


## 7. Benchmark classification on augmented dataset
7.1. To evaluate the new generated data on Gradient Boosting, run:
```bash
python benchmark_clf_main.py
```

7.2. As shown in STEP 2, you can change another classifier using the mentioned options.
Example:
```bash
python benchmark_clf_main.py --KNN_clf=True
```

*Options*
In addition to the options mentioned in STEP 1, we can pass the following:
- --labels: default="labels.csv" : refer to the labels generated

**Classification report of Random Forest tested on augmented data**
![rf_aug](https://github.com/user-attachments/assets/a56767b4-9718-47a1-973c-71dea1a55a13)

**Classification report of Extreme Gradient Boosting tested on augmented data**
![xgb_aug](https://github.com/user-attachments/assets/b0cede88-e92f-4de3-9b62-13078d07e382)

**Classification report of Gradient Boosting tested on augmented data**
![gb_aug](https://github.com/user-attachments/assets/d7d4f69c-b55f-46a7-89da-8f4e6bd5bdf0)

**Classification report of K Nearest Neighbor tested on augmented data**
![knn_aug](https://github.com/user-attachments/assets/6874711f-9524-47a5-90e7-390e4e112ac9)



# Comperative Studies
1.1. To train the Autoencoder (AE), choose one of the models by --model to RL-GAN or AE-DQN (required) and run:
```bash
 python comperativeStudies/AE_main.py --model RL-GAN --train --unaug_dataset
```
or
```bash
python comperativeStudies/AE_main.py --model AE-DQN --train --unaug_dataset
```

In case of RL-GAN, the proposed Generative Adversarial Network (GAN) is executed after the AE.

1.2. To test the Autoencoder (AE), run:
```bash
 python comperativeStudies/AE_main.py --model RL-GAN --unaug_dataset
```

2. Use the same script for benchmarking as shown in STEP 2 (AAE-DRL)
For example, to perform benchmark classification with GB, run:
```bash
python benchmark_clf_main.py --features=<path_to_dataset_generated_with_AE> --unaug_dataset
```
Specify the path to the dataset generated in the previous step (the default is AAE-DRL's dataset)

3. To pre-train the classifier implemented by one of the studies, run:
```bash
python comperativeStudies/Comp_classifier_main.py --model RL-GAN --train
```

4.1 To train TD3 (RL-GAN), run the same file implemented for AAE-DRL:
```bash
python DRL_main.py --train --RLGAN
```
To train DDQN (AE-DQN), run:
```bash
python comperativeStudies/DQN.py --train
```

4.2 To test TD3, run:
```bash
python DRL_main.py --train --RLGAN
```
To test DDQN, run:
```bash
python comperativeStudies/DQN.py --train
```

*Options*
In addition to the options mentioned in STEP4 (AAE-DRL), we can pass the following:
- --RLGAN: action="store_false" : switch from AAE-DRL to RL-GAN

5. To predict labels using the corresponding study classifier, run:
```bash
python comperativeStudies/Comp_classifier_main.py --label_gen --model RL-GAN   
```

6. To train the AE on augmented data, run:
```bash
python comperativeStudies/AE_main.py --model RL-GAN --train
```

7. Similarly, run the benchmark classification file to evaluate the augmented data:
```bash
python benchmark_clf_main.py --features=<path_to_dataset_generated_with_AE>
```
We assign another name in the --features option to avoid overwriting files.



# Paper
- AAE: Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., and Frey, B. (2015). [Adversarial autoencoders](https://arxiv.org/pdf/1511.05644). ArXiv.
- TD3: Fujimoto, S., van Hoof, H., and Meger, D. (2018). [Addressing function approximation error in actor-critic methods](https://arxiv.org/pdf/1802.09477). ArXiv.
- DDQN: van Hasselt, H., Guez, A., and Silver, D. (2015). [Deep reinforcement learning with double q-learning](https://arxiv.org/abs/1509.06461). ArXiv.
- TabNet: Arik S. O. and Pfister T. (2020). [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442). ArXiv.
- Attention Mechanism: Bahdanau D., Cho K. and Bengio Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473). ArXiv.
- Gradient Penalty: Gulrajani I., Ahmed F., Arjovsky M., Dumoulin V. and Courville A. (2017). [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028). ArXiv.
- Breiman L. (2001) [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324). Springer.
- Friedman J. (2000) [Greedy Function Approximation: A Gradient Boosting Machine](https://www.researchgate.net/publication/2424824_Greedy_Function_Approximation_A_Gradient_Boosting_Machine). The Annals of Statistics.
- Cheng T. and Guestrin C. (2016) [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). ArXiv.
- Cover T. and Hart P. (2006) [Nearest neighbor pattern classification](https://ieeexplore.ieee.org/document/1053964). IEEE



# Citations
- TD3
```bash
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}}
```

- Optuna
```bash
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}}
```

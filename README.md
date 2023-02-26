[Active Learning of Non-semantic Speech Tasks with Pretrained Models](http://arxiv.org/abs/2211.00119)
---
by Harlin Lee, Aaqib Saeed, Andrea L. Bertozzi @ 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

### Abstract
Pretraining neural networks with massive unlabeled datasets has become popular as it equips the deep models with a better prior to solve downstream tasks. However, this approach generally assumes that for downstream tasks, we have access to annotated data of sufficient size. In this work, we propose ALOE, a novel system for improving the data- and label-efficiency of non-semantic speech tasks with active learning. ALOE uses pre-trained models in conjunction with active learning to label data incrementally and learns classifiers for downstream tasks, thereby mitigating the need to acquire labeled data beforehand. We demonstrate the effectiveness of ALOE on a wide range of tasks, uncertainty-based acquisition functions, and model architectures. Training a linear classifier on top of a frozen encoder with ALOE is shown to achieve performance similar to several baselines that utilize the entire labeled data.

### Running the experiments
We provide an example of running an experiment with SpeechCommands dataset. 

#### 1. Installation
Install packages as follows:
```
pip3 install -r requirements.txt
```

#### 2. Extract features with a pretrained model: 
```
python3 fe.py
```
This will extract embeddings from train, validation and test splits of SpeechCommands and save them in 'features' directory.

#### 3. Run active learning with a linear model: 
```
python3 al.py --by_class --num_per_class=5 --iters=100 --seed=1
```
This will load the features along with correspoding labels and perform active learning for 100 iterations with initially 5 labeled example per class in a class-aware manner.

### Citation
```
@inproceedings{lee2023active,
  title={Active Learning of Non-semantic Speech Tasks with Pretrained Models},
  author={Lee, Harlin and Saeed, Aaqib and Bertozzi, Andrea L},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```

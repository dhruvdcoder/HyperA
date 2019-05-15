# Hyperbolic Attention

**Unofficial** pytorch implementation of the Neural Network components proposed in [Hyperbolic Neural Networks (NeurIPS)](https://papers.nips.cc/paper/7780-hyperbolic-neural-networks.pdf) plus experiments and implementations with attention mechanisms in Hyperbolic Space. 

# Usage

## 1. Setup Python paths

```
$ export PYTHONPATH=<repo_root_dir>
```

## 2. Install requirements

```
$ pip install -r requirements.txt
```

## 3. Train a model

Use the following comand to see the posible models and training configurations.

```
$ python hypernn/training.py --help

usage: training.py [-h] [-r ROOT_DIR] [-l {DEBUG,INFO,WARNING,ERROR}]
                   [--dataset {snli,multinli}] [--train_set TRAIN_SET]
                   [--val_set VAL_SET] [--test_set TEST_SET]
                   [--max_seq_len MAX_SEQ_LEN] [--vector_cache VECTOR_CACHE]
                   [--use_pretrained] [--emb_size EMB_SIZE]
                   [--emb_init_avg_norm EMB_INIT_AVG_NORM] [--freeze_emb]
                   [--resume_snapshot RESUME_SNAPSHOT] [--epochs EPOCHS]
                   [--batch_size BATCH_SIZE] [--save_dir SAVE_DIR]
                   [--save_every SAVE_EVERY] [--mode {test,train}]
                   [--device {cpu,gpu}] [--gpu GPU] [--dtype {double,float}]
                   [--mainlogdir MAINLOGDIR] [--experiment EXPERIMENT]
                   [--experiment_dir EXPERIMENT_DIR] [--tb_debug]
                   [--model {hconcatrnn,hdeepavg,haddrnn,hconcatgru,addrnnattn,hdeepavgattn}]
                   [--rnn {RNN,GRU}] [--combine_op {add,concat}]
                   [--hidden_dim HIDDEN_DIM] [--hyp_bias_lr HYP_BIAS_LR]
                   [--hyp_emb_lr HYP_EMB_LR] [--euc_lr EUC_LR]
                   [--print_every PRINT_EVERY] [--val_every VAL_EVERY]
                   [--debug_grad]

To level config for the HyperA project

optional arguments:
  -h, --help            show this help message and exit
  -r ROOT_DIR, --root_dir ROOT_DIR
                        Path to root directory for the project.
  -l {DEBUG,INFO,WARNING,ERROR}, --logging_level {DEBUG,INFO,WARNING,ERROR}
  --dataset {snli,multinli}
                        One of snli or multinli
  --train_set TRAIN_SET
                        file name of train dataset
  --val_set VAL_SET     file name of dev dataset
  --test_set TEST_SET   file name of test dataset
  --max_seq_len MAX_SEQ_LEN
                        Max sentence length. If you want to truncate
                        sentences. Default: None
  --vector_cache VECTOR_CACHE
                        filename for saving word embeddings cache
  --use_pretrained
  --emb_size EMB_SIZE   Has to be passed if --use_pretrained is not used
  --emb_init_avg_norm EMB_INIT_AVG_NORM
                        Average norm for uniformly initialized word
                        embs.Required if --use_pretrained is False
  --freeze_emb          Freeze embedding
  --resume_snapshot RESUME_SNAPSHOT
                        File where model snapshot is saved if resuming
                        training or doing test.
  --epochs EPOCHS       Number of epochs
  --batch_size BATCH_SIZE
  --save_dir SAVE_DIR   Dir where to save model snaps models are saved
  --save_every SAVE_EVERY
                        Checkpoint saving after how many iters
  --mode {test,train}
  --device {cpu,gpu}
  --gpu GPU             Used only if device is gpu
  --dtype {double,float}
                        double or float
  --mainlogdir MAINLOGDIR
                        Top level log dir. Logs for individual experiments are
                        subdirectories in this dir. (default: root_dir/logs)
  --experiment EXPERIMENT
                        name of the experiment. Use to create the logdir as
                        mainlogdir/experiment/
  --experiment_dir EXPERIMENT_DIR
                        if this is passed, --mainlogdir and --experiment are
                        ignored
  --tb_debug            Debug using tensorboard
  --model {hconcatrnn,hdeepavg,haddrnn,hconcatgru,addrnnattn,hdeepavgattn}
                        Pick the model to train
  --rnn {RNN,GRU}       Only used when using RNN based sentence encoder
  --combine_op {add,concat}
                        Method used to combine the reps of premise and hypo
                        when using model "haddrnn".
  --hidden_dim HIDDEN_DIM
  --hyp_bias_lr HYP_BIAS_LR
                        Learning rate for the bias parameters in the
                        hyperbolic space (used by RSGD)
  --hyp_emb_lr HYP_EMB_LR
                        Learning rate for the embedding parameters in the
                        hyperbolic space (used by RSGD)
  --euc_lr EUC_LR       Learning rate for layers parameterized by euclidean
                        params (used by Adam)
  --print_every PRINT_EVERY
                        Print training summary every
  --val_every VAL_EVERY
                        Run eval loop on dev data every val_every iterations
  --debug_grad          Pass this if debugging occurances of NaNs and Infs in
                        gradients

```

For instance, to train hyperbolic GRU, execute the following:

```
$ python hypernn/training.py --mainlogdir logs --experiment hyper_gru_5_5 --tb_debug --emb_size 5 --hidden_dim 5 --hyp_bias_lr 0.01 --hyp_emb_lr 0.2 --emb_init_avg_norm 0.0005 --epochs 30 --model haddrnn --rnn GRU --batch_size 512 --print_every 50 --device gpu
```

## 4. Extract predictions from the test set using a trained model

```
python hypernn/test.py --mode test --resume_snapshot .saved_models/hyper_gru_5_5/best_snapshot_devacc_56.85175751400917_devloss_0.9821267057980991__iter_59000_model.pt --experiment_dir logs/hyper_gru_5_5_2019-05-07T19-40-40
```
Supply, `--resume_snapshot` and `experiment_dir` appropriately from your training run. (Look at the console log of your training to find the paths to these two).

# Running baselines

Please see the README file in the Baseline folder.


## Running unit tests (NOT MANDATORY)

```
$ chmod +x run_tests.sh
$ ./run_tests.sh
```

# Contributors
1. [Dhruvesh Patel](https://github.com/dhruvdcoder)

2. [Prashant Ranjan](https://github.com/PrashantRanjan09)

3. [Praful Johri](https://github.com/prafuljohari)

4. [Rishabh Gupta](https://github.com/rishabh1694)

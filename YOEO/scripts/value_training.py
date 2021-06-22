"""
Train Value Function (eg) State-Value Function/Distribution, State-Action Value Function/Distribution)
"""
import os
import random
import argparse
import logging

import gin
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tfplot

import gym, d4rl

from YOEO.modules.utils import setup_logger, tqdm, write_gin_config

@gin.configurable(module=__name__)
def run(
    args,
    log_dir,
    seed,
    ########## gin controlled.
    Value,
    # dataset
    Dataset,
    # training loop
    num_updates,
    save_period, # in #updates
    eval_period, # in #updates
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    ####################
    # Define Network
    ####################
    value = Value()

    ########################
    # Define Dataset
    ########################
    D = Dataset(seed=seed)
    batch = iter(D.prepare_dataset())

    ########################
    # Prepare Update
    ########################
    update, report = value.prepare_update()

    ########################
    # Prepare Eval
    ########################
    def prepare_eval():
        train_t_0_batch = D.t_0_batch(type='train')
        valid_t_0_batch = D.t_0_batch(type='valid')

        train_eval_batch = D.eval_batch(type='train')
        valid_eval_batch = D.eval_batch(type='valid')

        def _predict(dataset):
            # (1) Calculate monte-carlo return (Note: use Bootstrapping for GT V calculation; no other
            # way we can do for cut-off trajectory) [B]
            # (2) Predict a value for a given state (or state-action pair) [B,num_samples]

            mc_V_s_t, est_V_s_t_samples = [], []
            for s_t, a_t, R_s_t, discount, s_last, a_last in dataset:
                mc_V_s_t.append(R_s_t + discount * np.mean(value(s_last,a_last).numpy(),axis=-1))
                est_V_s_t_samples.append(value(s_t,a_t).numpy())

            return np.concatenate(mc_V_s_t,axis=0),\
                np.concatenate(est_V_s_t_samples,axis=0)

        def eval_value(u):
            # For (s_0,a_0)
            train_mc_V_s_0, train_est_V_s_0_samples = _predict(train_t_0_batch)
            valid_mc_V_s_0, valid_est_V_s_0_samples = _predict(valid_t_0_batch)

            # Two histograms of mc_V_s_0 and est_V_s_0 should match.
            fig, axis = tfplot.subplots(1,2,figsize=(8, 3),sharex=True,sharey=True)
            _,bins,*_ = axis[0].hist(train_mc_V_s_0.ravel(),density=True,alpha=0.4,color='blue')
            axis[0].hist(train_est_V_s_0_samples.ravel(),bins=bins,density=True,alpha=0.4,color='red')
            axis[1].hist(valid_mc_V_s_0.ravel(),bins=bins,density=True,alpha=0.4,color='blue')
            axis[1].hist(valid_est_V_s_0_samples.ravel(),bins=bins,density=True,alpha=0.4,color='red')
            summary_writer.info('img',f'eval.{__name__}/hist_V_s_0',fig,u)
            plt.close(fig)

            # For random (s_t,a_t)
            train_mc_V_s_t, train_est_V_s_t_samples = _predict(train_eval_batch)
            valid_mc_V_s_t, valid_est_V_s_t_samples = _predict(valid_eval_batch)

            # |mc_V_s_t - mean(est_V_s_t)|
            # This is not a valid metric since est_V is an expectation while mc_V is a high-variance
            # sample. Therefore, the gap can be large even when trained V is valid when V^\pi(s_t) has a
            # large variance (for example, \pi is a set of policies)
            train_est_V_s_t, valid_est_V_s_t = np.mean(train_est_V_s_t_samples,axis=-1), np.mean(valid_est_V_s_t_samples,axis=-1)
            train_gap = np.mean(np.abs(train_mc_V_s_t - train_est_V_s_t))
            valid_gap = np.mean(np.abs(valid_mc_V_s_t - valid_est_V_s_t))
            summary_writer.info('raw',f'eval.{__name__}/train/mc_V-est_V',train_gap,u)
            summary_writer.info('raw',f'eval.{__name__}/valid/mc_V-est_V',valid_gap,u)

            fig, ax = tfplot.subplots(figsize=(4, 3))
            ax.plot(np.linspace(min(train_mc_V_s_t),max(train_mc_V_s_t),100),np.linspace(min(train_mc_V_s_t),max(train_mc_V_s_t),100),color='black')
            ax.scatter(train_mc_V_s_t,train_est_V_s_t,s=1.0,alpha=0.1,color='blue')
            ax.scatter(valid_mc_V_s_t,valid_est_V_s_t,s=1.0,alpha=0.1,color='red')
            summary_writer.info('img',f'eval.{__name__}/mc_V_and_est_V',fig,u)
            plt.close(fig)

            # quantile(mc_V;s_t,a_t) \in [0,1]
            # The mean of this value should be around 0.5, if distributional V is trained well.
            # The histogram over batch should be well-distributed.
            train_quantile = np.mean(train_est_V_s_t_samples <= train_mc_V_s_t[:,None],axis=-1)
            valid_quantile = np.mean(valid_est_V_s_t_samples <= valid_mc_V_s_t[:,None],axis=-1)
            summary_writer.info('raw',f'eval.{__name__}/train/mean_quantile',np.mean(train_quantile),u)
            summary_writer.info('raw',f'eval.{__name__}/valid/valid_quantile',np.mean(valid_quantile),u)
            summary_writer.info('histogram',f'eval.{__name__}/train/quantile_mc_V',train_quantile,u) # normal-like distribution would be nice.
            summary_writer.info('histogram',f'eval.{__name__}/valid/quantile_eval_V',valid_quantile,u) # normal-like distribution would be nice.

            # Qualitative examples (plot of value distribution + mc_V)
            # (1) a few randomly chosen (s_t,a_t)
            fig, axis = tfplot.subplots(2,5,figsize=(10,4))
            idxes = np.random.choice(len(train_mc_V_s_t),5)
            for ax, idx in zip(axis[0],idxes):
                y, *_ = ax.hist(train_est_V_s_t_samples[idx],density=True,alpha=0.4,color='red')
                ax.vlines(train_mc_V_s_t[idx],ymin=0.,ymax=max(y),color='blue')

            idxes = np.random.choice(len(valid_mc_V_s_t),5)
            for ax, idx in zip(axis[1],idxes):
                y, *_ = ax.hist(valid_est_V_s_t_samples[idx],density=True,alpha=0.4,color='red')
                ax.vlines(valid_mc_V_s_t[idx],ymin=0.,ymax=max(y),color='blue')

            summary_writer.info('img',f'eval.{__name__}/random_s_t/mc_V_and_est_V_sample',fig,u)
            plt.close(fig)

            # (2) distributionally worst-case (s_t,a_t); argmax |quantile(s_t,a_t,mc_V)-0.5|
            fig, axis = tfplot.subplots(2,5,figsize=(10,4))
            worst_idxes = np.argsort(np.abs(train_quantile - 0.5))[-5:]
            for ax, idx in zip(axis[0],worst_idxes):
                y, *_ = ax.hist(train_est_V_s_t_samples[idx],density=True,alpha=0.4,color='red')
                ax.vlines(train_mc_V_s_t[idx],ymin=0.,ymax=max(y),color='blue')

            worst_idxes = np.argsort(np.abs(valid_quantile - 0.5))[-5:]
            for ax, idx in zip(axis[1],worst_idxes):
                y, *_ = ax.hist(valid_est_V_s_t_samples[idx],density=True,alpha=0.4,color='red')
                ax.vlines(valid_mc_V_s_t[idx],ymin=0.,ymax=max(y),color='blue')

            summary_writer.info('img',f'eval.{__name__}/worst_s_t/mc_V_and_est_V_sample',fig,u)
            plt.close(fig)

        return eval_value

    eval_value = prepare_eval()

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

    ### Run
    try:
        for u in tqdm(range(num_updates)):
            _ = update(*next(batch))

            # log
            if (u+1) % 100 == 0:
                for name,item in report.items():
                    val = item.result().numpy()
                    summary_writer.info('raw',f'{__name__}/{name}',val,u+1)
                    item.reset_states()

            # eval
            if (u+1) % eval_period == 0:
                eval_value(u+1)

            # save
            if (u+1) % save_period == 0:
                value.save_weights(os.path.join(log_dir,f'value-{u+1}.tf'))

    except KeyboardInterrupt:
        pass

    value.save_weights(os.path.join(log_dir,f'value.tf'))

    logger.info('-------Gracefully finalized--------')
    logger.info('-------Bye Bye--------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--config_file',required=True, nargs='+')
    parser.add_argument('--config_params', nargs='*', default='')

    args = parser.parse_args()

    config_params = '\n'.join(args.config_params)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    if args.seed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_global_generator(tf.random.Generator.from_seed(args.seed))

    gin.parse_config_files_and_bindings(args.config_file, config_params)

    import YOEO.scripts.value_training
    YOEO.scripts.value_training.run(args,**vars(args))
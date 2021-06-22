"""
Train with pregenerated trajectories
"""
import os
import random
import logging
import argparse
from pathlib import Path

import gin
import numpy as np
import tensorflow as tf
import gym, d4rl
from matplotlib import pyplot as plt
import tfplot

from YOEO.modules.utils import setup_logger, tqdm, write_gin_config
from YOEO.modules.env_utils import interact, unroll_till_end
from YOEO.modules.d4rl_utils import get_nn

@gin.configurable
def prepare_eval(
    seed,
    algo,
    dataset,
    eval_env_id = None,
    eval_policies = ['pi'], # attribute name of algo
    num_eval_samples = 5000,
    num_eval_figures = 10,
):
    eval_env_id = dataset.env_id if eval_env_id is None else eval_env_id

    # preparation for eval_policy
    tests = {}
    for pi_name in eval_policies:
        test_env = gym.make(eval_env_id)
        test_env.seed(seed)

        test_pi = getattr(algo,pi_name)
        test_interact = interact(test_env, test_pi, stochastic=False)

        tests[pi_name] = (test_env,test_interact)

    def eval_policy(num_trajs):
        results = {}
        for pi_name, (test_env,test_interact) in tests.items():
            trajs = [unroll_till_end(test_interact) for _ in range(num_trajs)]

            Ts = [len(traj.states) for traj in trajs]
            returns = [np.sum(traj.rewards) for traj in trajs]

            if hasattr(test_env,'get_normalized_score'):
                norm_returns = [test_env.get_normalized_score(r) for r in returns]
            else:
                norm_returns = None

            results[pi_name] = (Ts, returns, norm_returns)

        return results

    if dataset is None:
        return eval_policy

    # preparation for eval_values
    nn, candidate_actions = get_nn(dataset.env_id)

    #train_eval_batch = dataset.eval_batch(type='train',batch_size=1000,max_num_samples=num_eval_samples)
    valid_eval_batch = dataset.eval_batch(type='valid',batch_size=1000,max_num_samples=num_eval_samples)

    t_0_batch = dataset.t_0_batch(batch_size=1000,type='valid',max_num_samples=-1)
    sample_states,sample_actions,*_ = next(iter(dataset.eval_batch(type='valid',batch_size=num_eval_figures)))

    q_value = algo.Qs[0]._build_Qs(algo.Qs)
    def _predict(dataset):
        mc_V, est_Q = [], []
        for s_t, a_t, R_s_t, discount, s_last, a_last in dataset:
            mc_V.append(R_s_t + discount * q_value(s_last,a_last).numpy())
            est_Q.append(q_value(s_t,a_t).numpy())

        return np.concatenate(mc_V,axis=0),np.concatenate(est_Q,axis=0)

    def eval_value():
        results = {}

        # Evaluate (1): MC Value v.s. Estimate Q value
        #train_mc_V, train_est_Q = _predict(train_eval_batch)
        valid_mc_V, valid_est_Q = _predict(valid_eval_batch)

        fig, ax = tfplot.subplots(1,1,figsize=(4, 3))
        #ax.plot(np.linspace(min(train_mc_V),max(train_mc_V),100),np.linspace(min(train_mc_V),max(train_mc_V),100),color='black')
        #ax.scatter(train_mc_V,train_est_Q,s=1.0,alpha=0.1,color='blue')
        ax.plot(np.linspace(min(valid_mc_V),max(valid_mc_V),100),np.linspace(min(valid_mc_V),max(valid_mc_V),100),color='black')
        ax.scatter(valid_mc_V,valid_est_Q,s=1.0,alpha=0.1,color='red')

        results['mc_V_and_est_V'] = fig

        # Evaluate (2): Q(s,a) vs Q(s,b) vs Q(s,\pi(s))
        fig, axis = tfplot.subplots(1,num_eval_figures,figsize=(2*num_eval_figures,1.5))
        for s, a, ax in zip(sample_states, sample_actions, axis):
            pi_a,_ = algo.pi(s[None],stochastic=False)
            np_pi_a,_ = algo.np_policy(s,stochastic=False)
            b = candidate_actions[np.array(nn.get_nns_by_vector(s,100))]
            random_b = np.random.uniform(low=-algo.pi.scale,high=algo.pi.scale,size=b.shape)

            q_sa = q_value(s[None],a[None]).numpy()[0]
            q_s_pi_a = q_value(s[None],pi_a).numpy()[0]
            q_s_np_pi_a = q_value(s[None],np_pi_a[None]).numpy()[0]
            q_s_b = q_value(np.repeat(s[None],len(b),axis=0),b).numpy()
            q_s_random_b = q_value(np.repeat(s[None],len(b),axis=0),random_b).numpy()

            y, *_ = ax.hist(q_s_b,density=True,alpha=0.4,color='green')
            ax.hist(q_s_random_b,density=True,alpha=0.4,color='red')

            ax.vlines(q_sa,ymin=0.,ymax=max(y),color='blue')
            ax.vlines(q_s_pi_a,ymin=0.,ymax=max(y),color='red')
            ax.vlines(q_s_np_pi_a,ymin=0.,ymax=max(y),color='green')

        results['Q_discern'] = fig

        # Evaluate (3): with (s_0,a_0); two histograms of mc_V_s_0 and est_V_s_0 should match.
        mc_V_s_0, est_Q_s_0 = _predict(t_0_batch)

        fig, ax = tfplot.subplots(1,1,figsize=(4, 3))
        _,bins,*_ = ax.hist(mc_V_s_0,density=True,alpha=0.4,color='blue')
        ax.hist(est_Q_s_0,bins=bins,density=True,alpha=0.4,color='red')

        results['hist_V_s_0'] = fig

        return results

    return eval_policy, eval_value

@gin.configurable(module=__name__)
def run(
    args,
    log_dir,
    seed,
    ########## gin controlled.
    Algo,
    Dataset,
    # training loop
    num_updates,
    log_period,
    save_period,
    run_period, # generate a single trajectory
    eval_period, # generate 100 trajectories
    num_bc_updates = None, # if you want to initialize a policy with BC, set this number. (
    **kwargs,
):
    # Define Logger
    setup_logger(log_dir,args)
    summary_writer = logging.getLogger('summary_writer')
    logger = logging.getLogger('stdout')

    chkpt_dir = Path(log_dir)/'chkpt'
    chkpt_dir.mkdir(parents=True,exist_ok=True)

    # Define Dataset
    dataset = Dataset(seed=seed)
    B = iter(dataset.prepare_dataset())

    # Define algorithm
    algo = Algo()
    update, report = algo.prepare_update(dataset=dataset)

    if num_bc_updates is not None:
        bc_update, bc_report = algo.pi.prepare_behavior_clone()

    _eval_policy, _eval_value = prepare_eval(seed, algo, dataset)

    def eval_policy(u, num_trajs):
        results = _eval_policy(num_trajs)

        for pi_name, (Ts, returns, norm_returns) in results.items():
            summary_writer.info('raw',f'eval.{__name__}/{pi_name}/mean_eps_length',np.mean(Ts),u)
            summary_writer.info('raw',f'eval.{__name__}/{pi_name}/mean_eps_return',np.mean(returns),u)
            summary_writer.info('raw',f'eval.{__name__}/{pi_name}/mean_eps_norm_return',np.mean(norm_returns),u)

            if len(returns) > 1:
                summary_writer.info('histogram',f'eval.{__name__}/{pi_name}/eps_return',returns,u)
                summary_writer.info('histogram',f'eval.{__name__}/{pi_name}/eps_norm_return',norm_returns,u)

    def eval_value(u):
        results = _eval_value()

        for name, fig in results.items():
            summary_writer.info('img',f'eval.{__name__}/{name}',fig,u)
            plt.close(fig)

    # write gin config right before run when all the gin bindings are mad
    write_gin_config(log_dir)

    ### Run
    try:
        for u in tqdm(range(num_updates)):
            if num_bc_updates is not None and u < num_bc_updates:
                s,a,*_ = next(B)
                bc_update(s,a)

                # log
                if (u+1) % log_period == 0:
                    for name,item in bc_report.items():
                        val = item.result().numpy()
                        summary_writer.info('raw',f'{__name__}/bc/{name}',val,u+1)
                        item.reset_states()

            else:
                update(*next(B))

                # log
                if (u+1) % log_period == 0:
                    for name,item in report.items():
                        val = item.result().numpy()
                        summary_writer.info('raw',f'{__name__}/{name}',val,u+1)
                        item.reset_states()

            # eval
            if (u+1) % run_period == 0 and (u+1) % eval_period != 0:
                eval_policy(u+1,num_trajs=5)

            if (u+1) % eval_period == 0:
                eval_value(u+1)
                eval_policy(u+1,num_trajs=20)

            # save
            if (u+1) % save_period == 0:
                algo.save_weights(str(chkpt_dir),u+1,with_Q=True)

        eval_value(u+1)
        eval_policy(u+1,num_trajs=100)

    except KeyboardInterrupt:
        pass

    algo.save_weights(log_dir,with_Q=True)

    logger.info('-------Gracefully finalized--------')
    logger.info('-------Bye Bye--------')

@gin.configurable
def eval(
    args,
    log_dir,
    seed,
    ################
    eval_env_id = None,
    eval_policies = ['np_policy'], #,'pi'],
    chkpt_it = None,
    num_trajs = 100,
    **kwargs
):
    algo = gin.query_parameter('YOEO.scripts.offline_rl.run.Algo').scoped_configurable_fn()
    algo.load_weights(log_dir if chkpt_it is None else os.path.join(log_dir,'chkpt'),chkpt_it,with_Q=True)

    eval_env_id = gin.query_parameter('%env_id') if eval_env_id is None else eval_env_id

    for pi_name in tqdm(eval_policies):
        tqdm.write(f'--------------------')
        tqdm.write(f'{pi_name}')
        tqdm.write(f'--------------------')

        _eval = prepare_eval(seed, algo, None, eval_env_id, [pi_name])

        results = []
        pbar = tqdm(range(num_trajs))
        for _ in pbar:
            T, r, norm_return = [e[0] for e in _eval(1)[pi_name]]
            results.append((T,r,norm_return))

            tqdm.write(f'{T}, {r}, {norm_return}')
            pbar.set_description(f'{np.mean([result[-1] for result in results])}')

        Ts, returns, norm_returns = zip(*results)
        tqdm.write(f'{np.mean(Ts)},{np.mean(returns)},{np.mean(norm_returns)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir',required=True)
    parser.add_argument('--config_file', nargs='*')
    parser.add_argument('--config_params', nargs='*', default='')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    config_params = '\n'.join(args.config_params)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.seed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_global_generator(tf.random.Generator.from_seed(args.seed))

    if not args.eval:
        gin.parse_config_files_and_bindings(args.config_file, config_params)

        import YOEO.scripts.offline_rl
        YOEO.scripts.offline_rl.run(args,**vars(args))
    else :
        gin.parse_config_files_and_bindings([os.path.join(args.log_dir,'config.gin')], config_params)

        import YOEO.scripts.offline_rl
        YOEO.scripts.offline_rl.eval(args,**vars(args))

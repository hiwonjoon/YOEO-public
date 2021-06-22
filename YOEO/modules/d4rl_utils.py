# D4RL Utils -- Dataset
import gin
import numpy as np
import tensorflow as tf

import gym, d4rl

from YOEO.modules.env_utils import Trajectory

def _parse_v0(env_id):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    obs, acs, rs, dones =\
        dataset['observations'], dataset['actions'], dataset['rewards'], dataset['terminals']

    def _parse(obs,actions,rewards,dones,trim_first_T,max_episode_steps):
        trajs = []
        start = trim_first_T
        while start < len(dones):
            end = start

            while end != 1000000 - 1 and end < len(dones) - 1 and \
                (not dones[end] and end - start + 1 < max_episode_steps):
                end += 1

            if dones[end]:
                # the trajectory ends normally.
                # since the next state will not be (should not be, actually) used by any algorithms,
                # we add null states (zero-states) at the end.

                traj = Trajectory(
                    states = np.concatenate([obs[start:end+1],np.zeros_like(obs[0])[None]],axis=0),
                    actions = actions[start:end+1],
                    rewards = rewards[start:end+1],
                    dones = dones[start:end+1].astype(np.bool),
                    frames = None,
                )

                assert np.all(traj.dones[:-1] == False) and traj.dones[-1]

            else:
                # episodes end unintentionally (terminate due to timeout, cut-off when concateante two trajectories, or etc).
                # since the next-state is not available, it drops the last action.

                traj = Trajectory(
                    states = obs[start:end+1],
                    actions = actions[start:end],
                    rewards = rewards[start:end],
                    dones = dones[start:end].astype(np.bool),
                    frames = None,
                )

                assert np.all(traj.dones == False)

            if len(traj.states) > 1: # some trajectories are extremely short in -medium-replay dataset (due to unexpected timeout caused by RLKIT); https://github.com/rail-berkeley/d4rl/issues/86#issuecomment-778566671
                trajs.append(traj)

            start = end + 1

        return trajs

    if env_id == 'halfcheetah-medium-replay-v0':
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)
    elif env_id == 'halfcheetah-medium-v0':
        trajs = _parse(obs,acs,rs,dones,899,env._max_episode_steps-1) # why env._max_episode_stpes - 1? it is questionable, but it looks a valid thing to do.
    elif env_id == 'halfcheetah-expert-v0':
        trajs = _parse(obs,acs,rs,dones,996,env._max_episode_steps-1)
    elif env_id == 'halfcheetah-medium-expert-v0':
        trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],899,env._max_episode_steps-1) + \
            _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],996,env._max_episode_steps-1)
    elif env_id == 'hopper-medium-v0':
        trajs = _parse(obs,acs,rs,dones,211,env._max_episode_steps)
    elif env_id == 'hopper-expert-v0':
        trajs = _parse(obs,acs,rs,dones,309,env._max_episode_steps-1)
    elif env_id == 'hopper-medium-expert-v0': # actually, expert + mixed
        trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],309,env._max_episode_steps-1) + \
            _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],0,env._max_episode_steps-1)
    elif env_id == 'walker2d-medium-v0':
        trajs = _parse(obs,acs,rs,dones,644,env._max_episode_steps)
    elif env_id == 'walker2d-expert-v0':
        trajs = _parse(obs,acs,rs,dones,487,env._max_episode_steps-1)
    elif env_id == 'walker2d-medium-expert-v0': # actually, expert + mixed
        trajs = _parse(obs[:1000000],acs[:1000000],rs[:1000000],dones[:1000000],644,env._max_episode_steps) + \
            _parse(obs[1000000:],acs[1000000:],rs[1000000:],dones[1000000:],487,env._max_episode_steps-1)
    elif env_id in ['halfcheetah-random-v0', 'walker2d-random-v0', 'hopper-random-v0', 'walker2d-medium-replay-v0', 'hopper-medium-replay-v0']:
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps-1)
    elif env_id in ['pen-expert-v0', 'hammer-expert-v0', 'door-expert-v0', 'relocate-expert-v0']:
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)
    elif env_id in ['door-human-v0','relocate-human-v0','hammer-human-v0']:
        trajs = _parse(obs,acs,rs,dones,0,1000)
        for traj in trajs:
            traj.dones[:] = False # this is philosophical decision; since its original env does not terminate, so 'done' in the human data does not meaning anything. I regard this information is given only as a trajectory separator.
    elif env_id in ['door-cloned-v0','relocate-cloned-v0','hammer-cloned-v0']:
        trajs = _parse(obs[:500000],acs[:500000],rs[:500000],dones[:500000],0,1000) + \
            _parse(obs[500000:],acs[500000:],rs[500000:],dones[500000:],0,env._max_episode_steps)
        for traj in trajs:
            traj.dones[:] = False # this is philosophical decision; since its original env does not terminate, so 'done' in the human data does not meaning anything. I regard this information is given only as a trajectory separator.
    elif env_id in ['pen-human-v0']:
        trajs = _parse(obs,acs,rs,np.zeros_like(dones),0,200)
        for traj in trajs:
            traj.dones[:] = False
    elif env_id in ['pen-cloned-v0']:
        trajs = _parse(obs[:250000],acs[:250000],rs[:250000],dones[:250000],0,200) + \
            _parse(obs[250000:],acs[250000:],rs[250000:],dones[250000:],0,env._max_episode_steps)
    else:
        trajs = _parse(obs,acs,rs,dones,0,env._max_episode_steps)

    return trajs

def get_nn(env_id,seed=0):
    import os
    from annoy import AnnoyIndex

    env = gym.make(env_id)
    dataset = D4RL_Dataset(env_id, seed=seed)
    actions = np.concatenate([traj.actions for traj in dataset.trajs],axis=0)

    s_index = AnnoyIndex(dataset.ob_dim[-1],metric='euclidean')
    s_index.set_seed(seed)
    s_index_fname = env.dataset_filepath+f'.s_index_v3.ann_seed{seed}'

    if os.path.exists(s_index_fname):
        s_index.load(s_index_fname)
    else:
        states = np.concatenate([traj.states[:-1] for traj in dataset.trajs],axis=0)
        assert len(actions) == len(states)

        print('build nn index...')

        for i,ob in enumerate(states):
            s_index.add_item(i,ob)
        s_index.build(30) # 30 trees
        s_index.save(s_index_fname)

        print('build nn inde complete!')

    return s_index, actions

@gin.configurable(module=__name__)
class D4RL_Dataset():
    @staticmethod
    def parse(env_id, drop_trailings):
        ## Parse the dataset into set of trajectories
        dataset = gym.make(env_id).get_dataset()
        obs, actions, rewards, terminals, timeouts =\
            dataset['observations'],\
            dataset['actions'],\
            dataset['rewards'],\
            dataset['terminals'],\
            dataset['timeouts']

        assert len(obs) == len(actions) == len(rewards) == len(terminals) == len(timeouts)
        N = len(obs)

        trajs = []

        start = 0
        while start < N:
            end = start
            while not (terminals[end] or timeouts[end]) and end < N-1:
                end += 1

            if timeouts[end] or (end == N-1 and not drop_trailings):
                # the trajectory ends due to some external cut-offs
                # since the next-state is not available, it drops the last action.

                traj = Trajectory(
                    states = obs[start:end+1],
                    actions = actions[start:end],
                    rewards = rewards[start:end],
                    dones = terminals[start:end].astype(np.bool),
                    frames = None,
                )

                assert np.all(traj.dones == False)

            elif terminals[end]:
                # the trajectory ends normally.
                # since the next state will not be (should not be, actually) used by any algorithms,
                # we add null states (zero-states) at the end.

                traj = Trajectory(
                    states = np.concatenate([obs[start:end+1],np.zeros_like(obs[0])[None]],axis=0),
                    actions = actions[start:end+1],
                    rewards = rewards[start:end+1],
                    dones = terminals[start:end+1].astype(np.bool),
                    frames = None,
                )

                assert np.all(traj.dones[:-1] == False) and traj.dones[-1]

            elif end == N-1 and drop_trailings:
                break

            else:
                assert False

            if len(traj.states) > 1: # some trajectories are extremely short in -medium-replay dataset (due to unexpected timeout caused by RLKIT); https://github.com/rail-berkeley/d4rl/issues/86#issuecomment-778566671
                trajs.append(traj)

            start = end + 1

        return trajs

    def __init__(self,env_id,gamma,drop_trailings=False,train_ratio=0.9,seed=None):
        self.env_id = env_id
        self.seed = seed

        self.env = gym.make(env_id)
        self.gamma = gamma

        if env_id.split('-')[-1] == 'v0':
            self.trajs = _parse_v0(self.env_id)
        else:
            self.trajs = self.parse(self.env_id, drop_trailings)

        self.ob_dim = self.trajs[0].states[0].shape
        self.ac_dim = self.trajs[0].actions[0].shape

        rng = np.random.default_rng(seed=seed) if seed is not None else np.random

        self.random_trajs = [(idx,self.trajs[idx]) for idx in rng.permutation(len(self.trajs))]
        self.train_trajs = self.random_trajs[:int(len(self.trajs)*train_ratio)]
        self.valid_trajs = self.random_trajs[int(len(self.trajs)*train_ratio):]

        ### Trajectory Info (s_0,a_0,R,discount,s_last,a_last)
        traj_info = []
        for idx,traj in enumerate(self.trajs):
            s_0,a_0 = traj.states[0], traj.actions[0]
            R = 0.

            if traj.dones[-1]:
                for r in traj.rewards[::-1]:
                    R = r + self.gamma * R 

                discount = 0.
                last_s = traj.states[-1]
                last_a = np.zeros_like(traj.actions[-1])
            else:
                for r in traj.rewards[-2::-1]:
                    R = r + self.gamma * R 

                discount = self.gamma ** len(traj.rewards[:-1])

                last_s = traj.states[-2]
                last_a = traj.actions[-1]
            
            traj_info.append((s_0,a_0,R,discount,last_s,last_a))
        
        self.b_s_0, self.b_a_0, self.b_R, self.b_discount, self.b_last_s, self.b_last_a = [tf.constant(e,tf.float32) for e in zip(*traj_info)]

    def _get_traj_dataset(self,type,shuffle=True):
        if type == 'all':
            idxes, trajs = zip(*self.random_trajs)
        elif type == 'train':
            idxes, trajs = zip(*self.train_trajs)
        elif type == 'valid':
            idxes, trajs = zip(*self.valid_trajs)
        else:
            assert False

        # Trajectory
        b_states, b_actions, b_rewards, b_dones, _ = zip(*trajs)

        b_states = tf.ragged.constant(b_states, inner_shape=self.ob_dim, dtype=tf.float32)
        b_actions = tf.ragged.constant(b_actions, inner_shape=self.ac_dim, dtype=tf.float32)
        b_rewards = tf.ragged.constant(b_rewards, inner_shape=(), dtype=tf.float32)
        b_dones = tf.ragged.constant(b_dones, inner_shape=(), dtype=tf.bool)

        dataset = tf.data.Dataset.from_tensor_slices(
            (b_states,b_actions,b_rewards,b_dones)
        )

        def _filter_sarsa(states,actions,rewards,dones):
            if dones[-1]: return True
            else: return len(states) >= 3

        dataset = dataset.filter(_filter_sarsa)

        if shuffle:
            dataset = dataset.shuffle(len(trajs),reshuffle_each_iteration=True)

        return dataset

    @gin.configurable(module=f'{__name__}.D4RL_Dataset')
    def t_0_batch(self,batch_size,type,max_num_samples=100):
        def saRsa_last(states,actions,rewards,dones):
            s,a = states[0], actions[0]
            if dones[-1]:
                R = tf.scan(lambda R, r: R * self.gamma + r, rewards, reverse=True)[0]
                discount = 0.
                # both last_s, last_a shouldn't matter.
                last_s = states[-1]
                last_a = tf.zeros_like(actions[-1])
            else:
                R = tf.scan(lambda R, r: R * self.gamma + r, rewards[:-1], reverse=True)[0]
                discount = self.gamma ** tf.cast(len(rewards[:-1]),tf.float32)

                last_s = states[-2]
                last_a = actions[-1]

            return (tf.cast(s,tf.float32),
                    tf.cast(a,tf.float32),
                    tf.cast(R,tf.float32),
                    tf.cast(discount,tf.float32),
                    tf.cast(last_s,tf.float32),
                    tf.cast(last_a,tf.float32))

        dataset = self._get_traj_dataset(type,shuffle=False)
        dataset = dataset.map(saRsa_last)
        dataset = dataset.take(max_num_samples)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    @gin.configurable(module=f'{__name__}.D4RL_Dataset')
    def eval_batch(self,batch_size,type,shuffle_size=50000,max_num_samples=5000):
        def saRsa_last(states,actions,rewards,dones):
            if dones[-1]:
                s = states[:-1]
                a = actions
                R = tf.scan(lambda R, r: R * self.gamma + r, rewards, reverse=True)
                discount = tf.zeros((tf.shape(s)[0],),tf.float32)
                # both last_s, last_a shouldn't matter. (becuase discount = 0, so it should be ignored)
                last_s = tf.repeat(states[-1][None],tf.shape(s)[0],axis=0)
                last_a = tf.repeat(tf.zeros_like(actions[-1])[None],tf.shape(s)[0],axis=0)
            else:
                s = states[:-2]
                a = actions[:-1]
                R = tf.scan(lambda R, r: R * self.gamma + r, rewards[:-1], reverse=True)
                discount = tf.math.pow(self.gamma * tf.ones((tf.shape(s)[0],),tf.float32), tf.range(tf.shape(s)[0],0,-1,tf.float32))
                last_s = tf.repeat(states[-2][None],tf.shape(s)[0],axis=0)
                last_a = tf.repeat(actions[-1][None],tf.shape(s)[0],axis=0)

            return tf.data.Dataset.from_tensor_slices((
                tf.cast(s,tf.float32),
                tf.cast(a,tf.float32),
                tf.cast(R,tf.float32),
                tf.cast(discount,tf.float32),
                tf.cast(last_s,tf.float32),
                tf.cast(last_a,tf.float32)))

        dataset = self._get_traj_dataset(type,shuffle=False)
        dataset = dataset.flat_map(saRsa_last)
        dataset = dataset.shuffle(shuffle_size,reshuffle_each_iteration=False)
        dataset = dataset.take(max_num_samples)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    @gin.configurable(module=f'{__name__}.D4RL_Dataset')
    def prepare_dataset(self,batch_size,shuffle_size=50000,include_na=True,repeat=True,take=-1,type='all',exclude_done=False,window_size=0):
        dataset = self._get_traj_dataset(type,shuffle=repeat)

        # Replay Buffer
        if include_na:
            def sarsa(states,actions,rewards,dones):
                if dones[-1]:
                    s = states[:-1]
                    a = actions
                    r = rewards
                    ns = states[1:]
                    na = tf.concat([actions[1:],tf.zeros_like(actions[:1])],axis=0)
                    d = dones
                else:
                    s = states[:-2]
                    a = actions[:-1]
                    r = rewards[:-1]
                    ns = states[1:-1]
                    na = actions[1:]
                    d = dones[:-1]

                return tf.data.Dataset.from_tensor_slices((
                    tf.cast(s,tf.float32),
                    tf.cast(a,tf.float32),
                    tf.cast(r,tf.float32),
                    self.gamma * (1.0-tf.cast(d,tf.float32)),
                    tf.cast(ns,tf.float32),
                    tf.cast(na,tf.float32)))

            dataset = dataset.flat_map(sarsa)
        else:
            def sars(states,actions,rewards,dones):
                return tf.data.Dataset.from_tensor_slices((
                    tf.cast(states[:-1],tf.float32),
                    tf.cast(actions,tf.float32),
                    tf.cast(rewards,tf.float32),
                    self.gamma * (1.0-tf.cast(dones,tf.float32)),
                    tf.cast(states[1:],tf.float32)))

            dataset = dataset.flat_map(sars)

        if exclude_done: # next-state is invalid (matter when the dataset is used to train transition dynamics)
            dataset = dataset.filter(lambda s,a,r,discount,*_: discount > 0.)
        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size,reshuffle_each_iteration=repeat)
        if repeat:
            dataset = dataset.repeat() # repeat indefinietly
        dataset = dataset.take(take)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if window_size > 0:
            dataset = tf.data.Dataset.zip((dataset,)*window_size)

        return dataset

if __name__ == "__main__":
    env_ids = [
        'hopper-random-v0',
        'hopper-medium-replay-v0',
        'hopper-medium-v0',
        'hopper-medium-expert-v0',
        'hopper-expert-v0',
        'walker2d-random-v0',
        'walker2d-medium-replay-v0',
        'walker2d-medium-v0',
        'walker2d-medium-expert-v0',
        'walker2d-expert-v0',
        'halfcheetah-random-v0',
        'halfcheetah-medium-replay-v0',
        'halfcheetah-medium-v0',
        'halfcheetah-medium-expert-v0',
        'halfcheetah-expert-v0',
        'hopper-random-v2',
        'hopper-medium-replay-v2',
        'hopper-medium-v2',
        'hopper-medium-expert-v2',
        'hopper-expert-v2',
        'walker2d-random-v2',
        'walker2d-medium-replay-v2',
        'walker2d-medium-v2',
        'walker2d-medium-expert-v2',
        'walker2d-expert-v2',
        'halfcheetah-random-v2',
        'halfcheetah-medium-replay-v2',
        'halfcheetah-medium-v2',
        'halfcheetah-medium-expert-v2',
        'halfcheetah-expert-v2',
        'kitchen-complete-v0',
        'kitchen-partial-v0',
        'kitchen-mixed-v0',
    ]

    # D4RL Dataset Sanity Check
    ## Check medium-expert dataset is just a concatenation of medium and expert.
    ## Then, it implies there is a tiny bug in qlearning_dataset / sequence_dataset.

    ## Fortunately, the previous problem in v0 seems like resolved.
    """
    def _check_medium_expert(env_type):
        me_dataset = gym.make(f'{env_type}-medium-expert-v2').get_dataset()
        m_dataset = gym.make(f'{env_type}-medium-v2').get_dataset()
        e_dataset = gym.make(f'{env_type}-expert-v2').get_dataset()
        assert len(m_dataset['observations']) == 1000000
        assert len(e_dataset['observations']) == 1000000
        print(len(me_dataset['observations']))
        #assert np.all(
        #    me_dataset['observations'],
        #    np.concatenate([m_dataset['observations'],e_dataset['observations']],axis=0)
        #)
    _check_medium_expert('hopper')
    _check_medium_expert('walker2d')
    _check_medium_expert('halfcheetah')
    """

    # Sanity Check - whether medium-expert is attached correctly
    """
    def _check_medium_expert_v2(env_type):
        me_dataset = D4RL_Dataset(f'{env_type}-medium-expert-v2')
        m_dataset = D4RL_Dataset(f'{env_type}-medium-v2',drop_trailings=True)
        e_dataset = D4RL_Dataset(f'{env_type}-expert-v2')
        assert len(me_dataset.trajs) == len(m_dataset.trajs) + len(e_dataset.trajs)
        assert np.all([np.all(a.rewards == b.rewards) for a,b in zip(me_dataset.trajs, m_dataset.trajs + e_dataset.trajs)])
    _check_medium_expert_v2('hopper')
    _check_medium_expert_v2('walker2d')
    _check_medium_expert_v2('halfcheetah')
    """

    # Sanity Check - whether the parse can be done correctly.
    # Check the initial state distribution holds
    """
    import matplotlib
    matplotlib.use('module://imgcat')
    from matplotlib import pyplot as plt
    for env_id in env_ids:
        dataset = D4RL_Dataset(env_id,gamma=0.99)
        env = gym.make(env_id)
        print('-----------------')
        print(env_id, len(dataset.trajs))
        print(dataset.trajs[0].states.shape, dataset.trajs[0].actions.shape)
        print(np.mean([len(traj.states) for traj in dataset.trajs]), np.std([len(traj.states) for traj in dataset.trajs]))
        print(np.mean([np.sum(traj.rewards) for traj in dataset.trajs]), np.std([np.sum(traj.rewards) for traj in dataset.trajs]))
        print('-----------------')
        ob_dim = min(dataset.trajs[0].states.shape[-1],10)
        fig,axis = plt.subplots(ob_dim,1,figsize=(3,8))
        # comparison group
        init_states = np.stack([env.reset() for _ in range(1000)],axis=0)
        dataset_init_states = np.stack([traj.states[0] for traj in dataset.trajs])
        for i in range(ob_dim):
            n,bins,*_ = axis[i].hist(init_states[:,i],density=True,alpha=0.5)
            axis[i].hist(dataset_init_states[:,i],density=True,alpha=0.5,bins=bins)
        fig.tight_layout()
        fig.show()
        fig.savefig(f'test_{env_id}.png')
        #input()
        plt.close(fig)
    """

    # Note: Due to RLKIT's weird behavior some of the trajectories are partial in -medium-replay datasets
    # Reason: When RLKIT collect the data, they thrown out a trailing trajectories, which is
    # longer than the requested data-length (default 1000). Therefore, there is a timeout even the
    # hopper env is "actually" timed-out.
    # (https://github.com/rail-berkeley/d4rl/issues/86#issuecomment-778566671)

    # Checkout Prepare Dataset Impl.
    """
    from tqdm import tqdm
    for env_id in tqdm(env_ids):
        dataset = D4RL_Dataset(env_id,drop_trailings=False)
        N = 0
        B = iter(dataset.prepare_dataset(64,include_na=False,repeat=False))
        for s,a,r,ns,done in B: N += len(s.numpy())
        assert N == np.sum([len(traj.rewards) for traj in dataset.trajs])
    """

    #from tqdm import tqdm
    #for _ in tqdm(range(10000)):
    #    s,a,r,ns,na,done = next(B)
    #    tqdm.write(f'{tf.shape(s)}, {tf.shape(a)}, {tf.shape(r)}, {tf.shape(ns)}, {tf.shape(na)}, {tf.shape(done)}')

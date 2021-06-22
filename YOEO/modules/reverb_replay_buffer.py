import os
import gin
import numpy as np
import tensorflow as tf
from filelock import FileLock

from YOEO.modules.d4rl_utils import D4RL_Dataset

class SARSA_Buffer(object):
    """
    Replay buffer with reverb library
    """

    def __init__(
        self,
        # Environment Settings
        gamma,
        ob_dim,
        ob_dtype,
        ac_dim,
        ac_dtype,
        # Replay Buffer Settings
        n_steps,
        add_trailing_n_step_items,
        max_size,
        priority_exponent, # prioritized sampling / if 0, uniform samoling
        remove_policy, # eviction policy / ['fifo', 'uniform']
        # Reader Settings
        min_size_to_sample,
        chkpt_path=None,
    ):
        import numpy as np
        import tensorflow as tf
        import reverb
        import tree

        _table_name = 'replay_buffer'
        # siganture of item --> will be converted with a data sample using Reader._convert function
        _signature = { 
            "traj_idx": tf.TensorSpec([], tf.int64),
            "t": tf.TensorSpec([], tf.int64),
            "s_t": tf.TensorSpec(ob_dim, ob_dtype),
            "a_t": tf.TensorSpec(ac_dim, ac_dtype),
            "r_[t,t')": tf.TensorSpec([n_steps], np.float32),
            "done_(t,t']": tf.TensorSpec([n_steps], np.bool), # the whole point of 'done' signal is to decide to utilize the s_t' or not / (after caculating R, this is the sole purpose)
            "s_t'": tf.TensorSpec(ob_dim, ob_dtype),
            "a_t'": tf.TensorSpec(ac_dim, ob_dtype),
        }

        _signature_shapes = tree.map_structure(lambda x: x.shape, _signature)
        _signature_dtypes = tree.map_structure(lambda x: x.dtype, _signature)

        rate_limiter = reverb.rate_limiters.MinSize(min_size_to_sample)

        if priority_exponent > 0.:
            sampler = reverb.selectors.Prioritized(priority_exponent=priority_exponent)
        else:
            sampler = reverb.selectors.Uniform()

        if remove_policy == 'fifo':
            remover = reverb.selectors.Fifo()
        elif remove_policy == 'uniform':
            remover = reverb.selectors.Uniform()
        else:
            assert False

        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=chkpt_path) if chkpt_path else None

        self.rb_server = reverb.Server(
            tables=[
                reverb.Table( # Table is about how you want to extract!
                    name=_table_name,
                    sampler=sampler,
                    remover=remover,
                    max_size=max_size,
                    rate_limiter=rate_limiter,
                    signature=_signature)
            ],
            port=None,
            checkpointer=checkpointer
        )

        _port = self.rb_server.port

        discount = tf.constant(
            [gamma**i for i in range(1,n_steps+1)], tf.float32) # g, g^2, ..., g^{n_steps} length = n_steps-1

        class Reader():
            @staticmethod
            def _convert(D):
                """
                return: (s,a,R,discount,ns,na)
                    (ns, na) can be zero array; but discount should be zero in that case!
                """
                # t' = t + n_step - 1

                traj_idx = D['traj_idx']
                t = D['t']

                s = D['s_t']
                a = D['a_t']
                ns = D["s_t'"]
                na = D["a_t'"]

                rewards = D["r_[t,t')"] #r_{t} r_{t+1} ... r_{t+n_step-1}; length = n_step
                dones = tf.cast(D["done_(t,t']"],tf.float32) #d_{t+1} d_{t+2} ... d_{t+n_step}; length = n_step

                # Note:
                # rewards can be 'nan', but it should be masked by dones via multiply_no_nan function.
                # It would be an internal error if reward is nan but done is not True.
                R = rewards[0] + tf.reduce_sum(discount[:-1] * tf.math.multiply_no_nan(rewards[1:],(1-dones[:-1])))

                return (traj_idx, t), (s, a, R, (1-dones[-1]) * discount[-1], ns, na)

            @staticmethod
            def get_dataset(server_addresses):
                def _make_dataset(server_address):
                    return reverb.TrajectoryDataset(
                        server_address=server_address,
                        table=_table_name,
                        shapes=_signature_shapes,
                        dtypes=_signature_dtypes,
                        max_in_flight_samples_per_worker=200, #batch_size*2,
                    )

                dataset = tf.data.Dataset.from_tensor_slices(server_addresses)
                dataset = dataset.interleave(
                    map_func=_make_dataset,
                    cycle_length=len(server_addresses),
                    num_parallel_calls=len(server_addresses),
                    deterministic=False)
                dataset = dataset.map(lambda item: Reader._convert(item.data),tf.data.experimental.AUTOTUNE)
                return dataset

            def __init__(self):
                self.server_address = f'localhost:{_port}'

            def get_update_priority_fn(self):
                if priority_exponent > 0.:
                    client = reverb.TFClient(self.server_address)
                    def update_priority(keys,priorities):
                        client.update_priorities(_table_name,keys,tf.cast(priorities,tf.float64))
                else:
                    update_priority = None

                return update_priority

        class Writer():
            # This `Class` will be pickled and constructed in another process!
            # (note that not an object / is there actual difference when it comes to pickling?)

            def __init__(self, priority_fn=None):
                import reverb
                self.client = reverb.Client(f'localhost:{_port}')
                self.writer = self.client.trajectory_writer(num_keep_alive_refs=n_steps+2)
                self.priority_fn = priority_fn

                self._traj_idx = None
                self._t = None

            def _write_tuple(self, action, reward, done, new_observation):
                self.writer.append({
                    'traj_idx':self._traj_idx,
                    't':self._t,
                    'action': ac_dtype(action),
                    'reward': np.float32(reward),
                    'done': np.bool(done),
                    'observation': ob_dtype(new_observation),
                })
                self._t += 1
                # a_{t-1}, r_{t-1}, d_{t}, s_{t} is recorded.

                if self._t >= n_steps+1:
                    self._write_item()

            def _write_item(self): #currently, (a_{t-1}, r_{t-1}, d_{t}, s_{t}) is the most recent item.
                n_step_traj = {
                    "traj_idx": self.writer.history['traj_idx'][-n_steps-2],
                    "t": self.writer.history['t'][-n_steps-2],
                    "s_t": self.writer.history['observation'][-n_steps-2], # s_{t-n-1}
                    "a_t": self.writer.history['action'][-n_steps-1], # a_{t-n-1}
                    "r_[t,t')": self.writer.history['reward'][-n_steps-1:-1], # r_{t-n-1} ... r_{t-2}: n rewards
                    "done_(t,t']": self.writer.history['done'][-n_steps-1:-1], # d_{t-n} ... d_{t-1}: n dones
                    "s_t'": self.writer.history['observation'][-2], # s_{t-1}
                    "a_t'": self.writer.history['action'][-1], # a_{t-1}
                }

                if priority_exponent > 0.:
                    _, (s,a,R,discount,ns,na) = Reader._convert({
                        key:item.numpy() for key, item in n_step_traj.items()
                    })
                    priority = self.priority_fn(s,a,R,discount,ns)
                else:
                    priority = 1.0

                self.writer.create_item( # the item (t_{t-n_steps} ... t+1) / (s_{t-n_steps}, s_{t+1}); in non-stacked form / a_t, r_t, done_t
                    table=_table_name,
                    priority=priority,
                    trajectory=n_step_traj)

            ####################################
            #### Exposed methods
            def init_episode(self,traj_idx,observation):
                self._traj_idx = traj_idx
                self._t = -1
                self._write_tuple(np.zeros(ac_dim,ac_dtype), np.nan, False, observation)

            def step(self,action,reward,done,new_observation):
                self._write_tuple(action,reward,done,new_observation)

            def end_episode(self,terminate):
                if terminate and add_trailing_n_step_items:
                    for _ in range(n_steps):
                        self._write_tuple(np.zeros(ac_dim,ac_dtype), np.nan, True, np.zeros(ob_dim,ob_dtype))
                return True

            def commit_episode(self):
                try:
                    self.writer.end_episode(timeout_ms=100)
                    self.writer.flush(timeout_ms=100)
                except (reverb.errors.DeadlineExceededError, RuntimeError) as e:
                    return False
                return True

        self.Reader = Reader
        self.Writer = Writer

    def get_writer_constructor(self):
        return self.Writer

    def get_reader_constructor(self):
        return self.Reader

@gin.configurable
class D4RL_Dataset_Reverb(D4RL_Dataset):
    def __init__(self,n_steps,save_chkpt=False,seed=None,**kwargs):
        super().__init__(seed=seed,**kwargs)

        self.n_steps = n_steps
        self.save_chkpt = save_chkpt and (self.seed is not None)

    @gin.configurable(module=f'{__name__}.D4RL_Dataset_Reverb')
    def prepare_dataset(self,batch_size,type='all',include_traj_info=False,window_size=0):
        if self.save_chkpt:
            chkpt_dir = f'{self.env.dataset_filepath}_{self.n_steps}_steps_seed_{self.seed}_{type}_reverb_chkpt_v2'

            lock = FileLock(f'{chkpt_dir}.lock')
            lock.acquire() # wait until chkpt is fully written

            am_i_writer = not os.path.exists(chkpt_dir)
        else:
            chkpt_dir = None
            am_i_writer = True

        self.B = SARSA_Buffer(
            self.gamma,
            self.ob_dim,np.float32,
            self.ac_dim,np.float32,
            self.n_steps,True,
            int(1e7),0.,'fifo',1,
            chkpt_dir
        )

        if am_i_writer:
            if type == 'all':
                trajs = self.random_trajs
            elif type == 'train':
                trajs = self.train_trajs
            elif type == 'valid':
                trajs = self.valid_trajs
            else:
                assert False

            writer = self.B.Writer()
            for traj_idx, (states, actions, rewards, dones, *_) in trajs:
                writer.init_episode(traj_idx,states[0])
                for a_t, r_t, d_t_plus_1, s_t_plus_1 in zip(actions,rewards,dones,states[1:]):
                    writer.step(a_t,r_t,d_t_plus_1,s_t_plus_1)
                writer.end_episode(dones[-1])
                writer.commit_episode()

            if self.save_chkpt:
                _ = writer.client.checkpoint()

        if self.save_chkpt:
            lock.release()

        reader = self.B.Reader()
        dataset = reader.get_dataset([reader.server_address])

        if include_traj_info:
            def _pack_traj_info(traj_info,sarsa):
                traj_idx,_ = traj_info

                #info = (self.b_s_0[traj_idx],self.b_a_0[traj_idx],self.b_R[traj_idx],self.b_discount[traj_idx],self.b_last_a[traj_idx])
                return traj_idx, sarsa

            dataset = dataset.map(_pack_traj_info, tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(lambda traj_info, sarsa: sarsa, tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if window_size > 0:
            dataset = tf.data.Dataset.zip((dataset,)*window_size)

        return dataset

if __name__ == "__main__":
    D = D4RL_Dataset_Reverb(n_steps=5,seed=1,env_id='hopper-medium-replay-v0',gamma=0.99)
    batch = D.prepare_dataset(10,type='train',include_traj_info=False,window_size=3)
    for i,(x,y,z) in enumerate(batch):
        print(x[2])
        print(y[2])
        print(z[2])
        print('---')

        if i >= 10:
            break
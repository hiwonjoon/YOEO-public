import os
import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from YOEO.modules.optimizer import ClipConstraint, AdamOptimizer

@gin.configurable(module=__name__)
class ActionValue(Model):
    def __init__(
        self,
        Net,
        build_target_net=False,
        name=None,
    ):
        super().__init__()

        self.net = Net(name=None if name is None else name)

        if build_target_net:
            self.target_net = Net(name=None if name is None else f'{name}_target')
            self.update_target(0.)
        else:
            self.target_net = self.net

    @property
    def trainable_variables(self):
        return self.net.trainable_variables

    @property
    def trainable_decay_variables(self):
        return self.net.decay_vars

    @tf.function
    def __call__(self,ob,ac,use_target=True): # default is using target network
        if use_target:
            return tf.squeeze(self.target_net((ob,ac)),axis=-1)
        else:
            return tf.squeeze(self.net((ob,ac)),axis=-1)

    @tf.function
    def update_target(self,τ):
        main_net_vars = sorted(self.net.trainable_variables,key = lambda v: v.name)
        target_net_vars = sorted(self.target_net.trainable_variables,key = lambda v: v.name)
        assert len(main_net_vars) > 0 and len(target_net_vars) > 0 and len(main_net_vars) == len(target_net_vars), f'{len(main_net_vars)} != {len(target_net_vars)}'

        for v_main,v_target in zip(main_net_vars,target_net_vars):
            v_target.assign(τ*v_target + (1-τ)*v_main)

    def td_target(self,R,ns,na,discount,policy_entropy,logp_na):
        return R + discount*(self(ns,na,use_target=True) - policy_entropy * logp_na)

    @gin.configurable(module=f'{__name__}.ActionValue')
    def td_loss(self,s,a,td_target,huber=False):
        q = self(s,a,use_target=False)

        if huber:
            abs_td_error = tf.abs(td_target - q)

            quad = tf.minimum(abs_td_error, 1.0)
            lin = (abs_td_error - quad)

            return tf.reduce_mean(0.5 * quad**2 + 1.0 * lin,axis=0)
        else:
            return 0.5 * tf.reduce_mean((q-td_target)**2,axis=0)

    ################ Static methods
    @staticmethod
    def _build_Qs(others,aggr='mean',**kwargs):
        if aggr == 'mean':
            def Q(ob,ac,use_target=True):
                return tf.add_n([other(ob,ac,use_target) for other in others]) / len(others)
        elif aggr == 'min':
            def Q(ob,ac,use_target=True):
                qs = tf.stack([other(ob,ac,use_target) for other in others],axis=-1)
                return tf.reduce_min(qs,axis=-1)
        else:
            assert False

        return Q

    @staticmethod
    def _build_td_target(td_targets):
        td_targets = tf.stack(td_targets,axis=1) #[B,#ensemble]
        return tf.reduce_min(td_targets,axis=1)

@gin.configurable
class ValidQbeta():
    """
    - Valid Q^\beta network
    - Supervised via Y
        - Specifically, Q(s,\cdot) larger than Y@q will be penalized to have smaller value than that.
    """
    def __init__(
        self,
        num_ensembles,
        ActionValue,
        Policy,
        build_np_policy=True,
        np_policy_params={
            'build_qs_kwargs': {'aggr':'min'},
            'prefilter_type': 'state',
            'num_candidates': 100,
        },
    ):
        self.Qs = [ActionValue(name=f'Q{i}') for i in range(num_ensembles)]
        self.pi = Policy()

        if build_np_policy:
            from YOEO.modules.d4rl_utils import get_nn

            env_id = gin.query_parameter('%env_id')
            nn, actions = get_nn(env_id)

            Q = ActionValue._build_Qs(self.Qs,**np_policy_params['build_qs_kwargs'])

            if np_policy_params['prefilter_type'] == 'random':
                def _action_candidates(ob):
                    return actions[np.random.choice(len(actions),np_policy_params['num_candidates'],replace=False)]
            elif np_policy_params['prefilter_type'] == 'state':
                def _action_candidates(ob):
                    candidates = nn.get_nns_by_vector(ob,np_policy_params['num_candidates'])
                    return actions[np.array(candidates)]
            else:
                assert False

            @tf.function
            def _pick_best_action(ob,candidate_acs):
                q = Q(tf.repeat(ob[None],len(candidate_acs),axis=0),candidate_acs,use_target=False)
                return candidate_acs[tf.argmax(q)]

            def np_policy(ob,stochastic):
                candidates = _action_candidates(ob)
                best_acs = _pick_best_action(ob,candidates)
                return best_acs, None

            self.np_policy = np_policy


    @gin.configurable(module=f'{__name__}.ValidQbeta')
    def prepare_update(
        self,
        use_different_batch,
        Optimizer,
        # Conservative loss (Q upper-bound by Y) related
        ValueDistribution,
        ValueDistribution_chkpt,
        independent_ensembles, # if True, pair each Q to each separate Y.
        q_quantile,
        num_b:int, # number of non-beta actions for regularization
        q_sb_alpha,
        q_sb_ub_quantile,
        q_sb_ub_temp, # temperature for soft-max,
        q_pi_alpha,
        q_pi_ub_quantile,
        q_pi_ub_temp, # temperature for soft-max
        # TD-3 type Q-smoothing
        action_noise_sigma,
        action_noise_clip,
        **kwargs,
    ):
        tf_rg = tf.random.get_global_generator()

        report = {
            'Qbeta/td_loss':tf.keras.metrics.Mean(),
            'Qbeta/q_sb_ub_loss':tf.keras.metrics.Mean(),
            'Qbeta/q_pi_ub_loss':tf.keras.metrics.Mean(),
            'Qbeta/gap/q_sb_ub-max_q_sb':tf.keras.metrics.Mean(),
            'Qbeta/gap/q_pi_ub-q_pi':tf.keras.metrics.Mean(),
            'Qbeta/gap/q_sa-q_pi':tf.keras.metrics.Mean(),
        }

        Y = ValueDistribution()
        Y.load_weights(ValueDistribution_chkpt)

        if independent_ensembles:
            Ys = Y.ensembles
        else:
            Ys = [Y] * len(self.Qs)

        ### Prepare pi Update
        pi_update, pi_report = self.pi.prepare_update(self.Qs)
        for key,item in pi_report.items():
            report[f'Qbeta/pi/{key}'] = item

        ### Prepare Q Update
        q_optimizers = [Optimizer(Q.trainable_variables,Q.trainable_decay_variables) for Q in self.Qs]

        def _q_update(q_optimizer,Q,Y,batch):
            s,a,R,discount,ns,na = batch
            B = tf.shape(s)[0]

            td_target = R + discount*Y.quantile(ns,na,q_quantile,use_target=False)

            q_sa = Q(s,a,use_target=False)
            q_sb_ub = Y.quantile(s,a,q_sb_ub_quantile,use_target=False)
            q_pi_ub = Y.quantile(s,a,q_pi_ub_quantile,use_target=False)

            with tf.GradientTape() as tape:
                ## TD-Loss
                td_loss = Q.td_loss(s,a,td_target)

                ## Conservative Loss (1) - Q(s,b) upper bound
                non_beta_actions = tf_rg.uniform(tf.concat([[num_b],tf.shape(a)],axis=0),-self.pi.scale,self.pi.scale)
                q_sb = Q(tf.repeat(s[None],tf.shape(non_beta_actions)[0],axis=0),non_beta_actions,use_target=False)

                soft_max_q_sb = q_sb_ub_temp * tf.math.reduce_logsumexp(
                    (tf.concat([q_sb_ub[None],q_sb],axis=0) - q_sb_ub[None]) / q_sb_ub_temp,
                    axis=0)
                q_sb_ub_loss = tf.reduce_mean(soft_max_q_sb,axis=0) #maximum Q(s,b) <= Q_sb upper bound

                ## Conservative Loss (2) - Q(s,pi(s)) upper bound
                pi_s,_ = self.pi.action(s,stochastic=False)

                action_noise = tf_rg.normal(tf.concat([[num_b],tf.shape(pi_s)],axis=0),stddev=action_noise_sigma)
                action_noise = tf.clip_by_value(action_noise,-action_noise_clip,action_noise_clip)

                adv_actions = pi_s[None] + action_noise
                adv_actions = tf.clip_by_value(adv_actions,-self.pi.scale,self.pi.scale)

                q_pi = Q(tf.repeat(s[None],tf.shape(adv_actions)[0],axis=0), adv_actions, use_target=False)
                soft_max_q_pi = q_pi_ub_temp * tf.math.reduce_logsumexp(
                    (tf.concat([q_pi_ub[None],q_pi],axis=0) - q_pi_ub[None]) / q_pi_ub_temp,
                    axis=0)

                q_pi_ub_loss = tf.reduce_mean(soft_max_q_pi,axis=0) #maximum Q(s,b) <= Q_sb upper bound

                # Sum of loss
                loss = q_sb_alpha * q_sb_ub_loss + q_pi_alpha * q_pi_ub_loss + td_loss

                report['Qbeta/td_loss'](td_loss)
                report['Qbeta/q_sb_ub_loss'](q_sb_ub_loss)
                report['Qbeta/q_pi_ub_loss'](q_pi_ub_loss)

            report['Qbeta/gap/q_sb_ub-max_q_sb'](tf.reduce_mean(q_sb_ub - tf.reduce_max(q_sb,axis=0),axis=0))
            report['Qbeta/gap/q_pi_ub-q_pi'](tf.reduce_mean(q_pi_ub - tf.reduce_max(q_pi,axis=0),axis=0))
            report['Qbeta/gap/q_sa-q_pi'](tf.reduce_mean(q_sa - tf.reduce_max(q_pi,axis=0),axis=0))

            q_optimizer.minimize(tape,loss)

        @tf.function
        def update(*e_data):
            if use_different_batch == False:
                e_data = [e_data] * len(self.Qs)

            for q_optimizer,Q,Y,batch in zip(q_optimizers,self.Qs,Ys,e_data):
                _q_update(q_optimizer,Q,Y,batch)

            s,*_ = e_data[0]
            pi_update(s)

        return update, report

    def save_weights(self,log_dir,it=None,with_Q=True):
        self.pi.save_weights(os.path.join(log_dir,'pi.tf' if it is None else f'pi-{it}.tf'))
        if with_Q:
            for i, Q in enumerate(self.Qs):
                Q.save_weights(os.path.join(log_dir,f'Q{i}.tf' if it is None else f'Q{i}-{it}.tf'))

    def load_weights(self,log_dir,it=None,with_Q=True):
        self.pi.load_weights(os.path.join(log_dir,'pi.tf' if it is None else f'pi-{it}.tf'))
        if with_Q:
            for i, Q in enumerate(self.Qs):
                Q.load_weights(os.path.join(log_dir,f'Q{i}.tf' if it is None else f'Q{i}-{it}.tf'))


@gin.configurable
class Qbeta(ValidQbeta):
    """
    - Ablation Study (just Q^\beta)
    """

    @gin.configurable(module=f'{__name__}.Qbeta')
    def prepare_update(
        self,
        use_different_batch,
        Optimizer,
        polyak_coeff,
        **kwargs,
    ):
        tf_rg = tf.random.get_global_generator()
        report = {
            'Qbeta/td_loss':tf.keras.metrics.Mean(),
        }

        ### Prepare pi Update
        pi_update, pi_report = self.pi.prepare_update(self.Qs)
        for key,item in pi_report.items():
            report[f'Qbeta/pi/{key}'] = item

        ### Prepare Q Update
        q_optimizers = [Optimizer(Q.trainable_variables,Q.trainable_decay_variables) for Q in self.Qs]

        def _q_update(q_optimizer,Q,batch):
            s,a,R,discount,ns,na = batch

            td_target = R + discount * Q(ns,na,use_target=True)
            with tf.GradientTape() as tape:
                ## TD-Loss
                loss = Q.td_loss(s,a,td_target)

            q_optimizer.minimize(tape,loss)
            report['Qbeta/td_loss'](loss)

        @tf.function
        def update(*e_data):
            if use_different_batch == False:
                e_data = [e_data] * len(self.Qs)

            for q_optimizer,Q,batch in zip(q_optimizers,self.Qs,e_data):
                _q_update(q_optimizer,Q,batch)
                Q.update_target(polyak_coeff)

            s,*_ = e_data[0]
            pi_update(s)

        return update, report

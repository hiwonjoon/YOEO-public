import gin
import tensorflow as tf
from tensorflow.keras import Model

import YOEO.modules.optimizer
from YOEO.modules.quantile_regression import QuantileNetwork

@gin.configurable
class Y(Model):
    def __init__(
        self,
        Psi, # Feature Generator; R^(|s|) -> R^|d|
        Phi, # Tau feature generator; [0,1] -> R^|d|
        F, # MLP generating final action-value prediction Z@tau; R^|d| -> R
        build_target_net=False,
    ):
        super().__init__()

        self.psi = Psi()
        self.phi = Phi()
        self.f = F()

        self.qn = QuantileNetwork(self.psi,self.phi,self.f)

        if build_target_net:
            self.target_psi = Psi()
            self.target_phi = Phi()
            self.target_f = F()

            self.update_target(0.)

            self.target_qn = QuantileNetwork(self.target_psi,self.target_phi,self.target_f)
        else:
            self.target_qn = self.qn

    def quantile(self,s,a,q,use_target=False):
        qn = self.target_qn if use_target else self.qn
        return qn.quantile(s,q)

    def quantile_batch(self,s,a,q,use_target=False):
        qn = self.target_qn if use_target else self.qn
        return qn.call(s,q)

    def CDF(self,s,a,v,use_target=False,num_samples=100):
        qn = self.target_qn if use_target else self.qn
        _, alpha = qn.CDF(s,v,num_samples)
        return alpha

    def call(self,s,a,num_samples=100,use_target=False):
        # Return a value distribution given state
        # (action will be ignored since this is state-value distribution)
        # [B,num_samples]
        qn = self.target_qn if use_target else self.qn

        return qn.sample(s,num_samples)[1]

    @property
    def trainable_variables(self):
        return self.psi.trainable_variables + self.phi.trainable_variables + self.f.trainable_variables

    @property
    def trainable_decay_variables(self):
        return self.psi.decay_vars + self.phi.decay_vars + self.f.decay_vars

    @tf.function
    def update_target(self,polyak):
        for net, target_net in [
            (self.psi,self.target_psi),
            (self.phi,self.target_phi),
            (self.f,self.target_f)
        ]:
            main_net_vars = sorted(net.trainable_variables,key = lambda v: v.name)
            target_net_vars = sorted(target_net.trainable_variables,key = lambda v: v.name)
            assert len(main_net_vars) > 0 and len(target_net_vars) > 0 and len(main_net_vars) == len(target_net_vars), f'main: {len(main_net_vars)} target: {len(target_net_vars)}'

            for v_main,v_target in zip(main_net_vars,target_net_vars):
                v_target.assign(polyak*v_target + (1-polyak)*v_main)

    @gin.configurable(module=f'{__name__}.Y')
    def prepare_update(
        self,
        N_i,
        N_j,
        kappa,
        polyak,
        Optimizer=YOEO.modules.optimizer.AdamOptimizer,
    ):
        optimizer = Optimizer(self.trainable_variables,self.trainable_decay_variables)

        @tf.function
        def update(s,a,r,discount,ś,á):
            target_Y = r[:,None] + discount[:,None]*(self.target_qn.sample(ś,N_j)[1]) #[B,N_j]

            with tf.GradientTape() as tape:
                # TD loss
                L = self.qn.huber_quantile_loss(s,target_Y,N_i,kappa)
                loss = tf.reduce_mean(L)

            optimizer.minimize(tape,loss)

            if self.qn != self.target_qn:
                self.update_target(polyak)

            return loss

        return update, optimizer.reports

@gin.configurable
class Y_Ensemble(Model):
    def __init__(
        self,
        num_ensembles
    ):
        super().__init__()

        self.ensembles = [Y() for _ in range(num_ensembles)]

    @tf.function()
    def call(self,s,a,num_samples=100,use_target=False):
        return tf.concat([Y(s,a,max(num_samples//len(self.ensembles),1),use_target) for Y in self.ensembles],axis=-1)

    def quantile(self,s,a,q,use_target=False):
        return tf.add_n([Y.quantile(s,a,q,use_target) for Y in self.ensembles]) / len(self.ensembles)

    @tf.function
    def update_target(self,polyak):
        for Y in self.ensembles:
            Y.update_target(polyak)

    @gin.configurable(module=f'{__name__}.Y_Ensemble')
    def prepare_update(
        self,
        use_different_batch
    ):
        update_fns, reports = zip(*[Y.prepare_update() for Y in self.ensembles])

        agg_report = {key:type(item)()  for report in reports for key, item in report.items()}

        @tf.function
        def update_fn(*e_data):
            losses = []

            if use_different_batch == False:
                e_data = [e_data] * len(self.ensembles)

            for update_fn, batch in zip(update_fns, e_data):
                losses.append(update_fn(*batch))

            # Aggregate log data
            for key, item in agg_report.items():
                for report in reports:
                    item(report[key].result())
                    report[key].reset_states()

            return tf.add_n(losses)

        return update_fn, agg_report

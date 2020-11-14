import torch

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

@wrap_experiment
def ppo_cartpole(ctxt=None, seed=1):
    """Train PPO with CartPole-v0 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)
    env = GarageEnv(env_name='CartPole-v0')

    runner = LocalRunner(snapshot_config=ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               max_path_length=100,
               discount=0.99,
               center_adv=False)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=10000)


ppo_cartpole(seed=1)
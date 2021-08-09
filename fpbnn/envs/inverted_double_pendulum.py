from data.mujoco import Mujoco
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv as Env


class InvertedDoublePendulum(Mujoco):
    def __init__(self, std_capsule=0.1, **kwargs):
        super().__init__(Env(), **kwargs)
        self.std_capsule = std_capsule

    def parse_xml(self):
        self.parse_capsule()


if __name__ == '__main__':
    env_ = InvertedDoublePendulum()
    env_.render_random_policy()

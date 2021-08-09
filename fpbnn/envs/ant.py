from gym.envs.mujoco.ant import AntEnv as Env

from envs.mujoco import Mujoco


class Ant(Mujoco):
    def __init__(self, std_capsule=0.3, **kwargs):
        super().__init__(Env(), **kwargs)
        self.std_capsule = std_capsule

    def parse_xml(self):
        self.parse_capsule()


if __name__ == '__main__':
    env_ = Ant(seed=123)
    env_.render_random_policy()

from data.mujoco.mujoco import Mujoco
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as Env


class HalfCheetah(Mujoco):
    """
    Randomized Half Cheetah environment
    """

    def __init__(self, std_stiffness=0.1, std_capsule=0.1, **kwargs):
        super().__init__(Env(), **kwargs)
        self.std_stiffness = std_stiffness
        self.std_capsule = std_capsule

    def parse_xml(self):
        """
        Check the parent class for documentation
        """
        self.parse_capsule()
        self.parse_stiffness()


if __name__ == '__main__':
    env_ = HalfCheetah(seed=123)
    env_.render_random_policy()

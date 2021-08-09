from data.mujoco.mujoco import Mujoco
from gym.envs.mujoco.swimmer import SwimmerEnv


class Swimmer(Mujoco):
    def __init__(self, std_capsule=0.1, std_viscosity=0.1, **kwargs):
        super().__init__(SwimmerEnv(), **kwargs)
        self.std_capsule = std_capsule
        self.std_viscosity = std_viscosity

    def parse_xml(self):
        self.parse_capsule()
        self.parse_viscosity()


if __name__ == '__main__':
    env_ = Swimmer()
    env_.render_random_policy()

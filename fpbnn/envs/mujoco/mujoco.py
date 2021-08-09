"""
Base class for MuJoCo environments.
"""
import os.path
from multiprocessing import Pool
from xml.etree import ElementTree as ET

import mujoco_py
import numpy as np

from envs.regression import RegressionDataset


class Mujoco(RegressionDataset):
    def __init__(self, env, processes=16, **kwargs):
        super().__init__(**kwargs)
        # env.frame_skip = 1

        # load environment
        self.std_capsule = None
        self.env = env
        self.processes = processes

        # define state and action spaces
        self.state_space_size = self.env.sim.model.nq + self.env.sim.model.nv
        self.action_space_size = np.prod(self.env.action_space.shape)

        self.qpos_size = self.env.model.nq
        self.qvel_size = self.env.model.nv

        # define input and outpu dimensions
        self.input_dim = self.state_space_size + self.action_space_size
        self.output_dim = self.state_space_size

        self.state_range = np.stack([self.env.observation_space.low, self.env.observation_space.high]).tolist()
        self.action_range = np.stack([self.env.action_space.low, self.env.action_space.high]).tolist()

        self.env.reset()

        path = f'{os.path.dirname(__file__)}/assets/{self.name}.xml'
        with open(path, 'r') as f:
            xml_string = f.read()
        self.root = ET.fromstring(xml_string)

    def sample_function(self):
        def g(states):
            """
            states: (m, d + a)
            next_states: (m, d)
            """
            with Pool(self.processes) as pool:
                next_states = pool.map(self.step, states)
            # next_states = list(map(self.step, states))
            return np.stack(next_states)

        return g

    def step(self, state: np.ndarray) -> np.ndarray:
        self.sample_new_env()
        state, action = np.split(state, [self.state_space_size])
        self.env.set_state(*np.split(state, [self.qpos_size]))
        self.env.step(action)
        self.env.close()
        return self.env.state_vector().astype(np.float32)

    def get_images_from_random_env(self, h=3, w=4, res=800, n_steps=0, path='./plots/'):
        """
        Args:
            h: height
            w: width
            res: resolution (higher the better)
            n_steps: number of steps to run from initial position
        """
        from PIL import Image

        h, w = h * res, w * res
        np.random.seed(self.seed)

        self.sample_new_env()
        self.run_random_steps(n_steps)

        # render
        viewer = mujoco_py.MjViewerBasic(self.env.sim)
        viewer.render()

        # viewer.cam.fixedcamid = 0
        if self.name in ['half_cheetah', 'ant', 'swimmer']:
            camera_name = 'track'
        else:
            camera_name = None
        image_array = self.env.sim.render(height=h, width=w, camera_name=camera_name)
        self.env.close()

        # specify crop dimensions
        if self.name == 'inverted_double_pendulum':
            crop_dims = 0.3 * w, 0.3 * h, 0.7 * w, 0.6 * h
        elif self.name == 'swimmer':
            crop_dims = 0.01 * w, 0.01 * h, 0.99 * w, 0.99 * h
        elif self.name == 'half_cheetah':
            crop_dims = 0.01 * w, 0.3 * h, 0.99 * w, 0.99 * h
        else:
            raise NotImplemented(f'Crop dimensions for {self.name} are not implemented')

        # load, crop & save
        image = Image.fromarray(image_array).transpose(Image.FLIP_TOP_BOTTOM)
        image = image.crop(crop_dims)
        image.save(path + self.name + '_' + str(self.seed) + ".jpg")

    def run_random_steps(self, n_steps: int):
        for _ in range(n_steps):
            action = self.env.action_space.sample()
            self.env.step(action)

    def render_random_policy(self, n_episodes=10, episode_length=100):
        self.sample_new_env()
        self.env.frame_skip = 1
        for i_episode in range(n_episodes):
            self.env.reset()
            for t in range(episode_length):
                self.env.render()
                action = self.env.action_space.sample()
                self.env.step(action)
        self.env.close()

    def __str__(self):
        res = f'{self.__class__.__name__}\n\t' \
              f'state space size: {self.state_space_size}\n\t' \
              f'action space size: {self.action_space_size}\n\t' \
              f'feasible state space: {self.state_range}\n\t' \
              f'feasible action space: {self.action_range}'
        return res

    def sample_new_env(self):
        self.parse_xml()
        xml_string = ET.tostring(self.root, encoding="unicode")
        model = mujoco_py.load_model_from_xml(xml_string)
        self.env.sim = mujoco_py.MjSim(model)  # 1e-3 seconds (expensive)
        self.env.frame_skip = 1
        # return self.env

    def parse_xml(self):
        """
        For all children classes, include a series of changes you wish to make to the xml file, e.g.
        >>> self.parse_capsule()
        """
        raise NotImplemented

    def parse_capsule(self):
        # change geometry properties
        for child in self.root.iter('geom'):
            if 'type' in child.attrib:
                if child.attrib['type'] == 'capsule':
                    x = child.attrib['size']  # read value as string
                    x = [float(_) for _ in x.split(' ')]  # convert to list of floats
                    m = np.log(x) - self.std_capsule ** 2 / 2  # compute mean of lognormal
                    s = self.std_capsule  # compute mean of lognormal
                    x = self.random_state.lognormal(m, s)  # sample new values
                    x = ' '.join(map(str, x))  # convert to string
                    child.attrib['size'] = x  # load to xml

    def parse_stiffness(self):
        # change joint stiffness
        for child in self.root.iter('joint'):
            if 'type' in child.attrib:
                if child.attrib['type'] == 'hinge':
                    s = child.attrib['stiffness']  # read value as string
                    if s != '0':
                        s = [float(_) for _ in s.split(' ')]  # convert to list of floats
                        s = np.log(s) - self.std_capsule ** 2 / 2  # compute mean of lognormal
                        s = self.random_state.lognormal(s, self.std_capsule)  # sample new values
                        s = ' '.join(map(str, s))  # convert to string
                        child.attrib['stiffness'] = s  # load to xml

    def parse_viscosity(self):
        # change viscosity for swimmer 
        for child in self.root.iter('option'):
            s = child.attrib['viscosity']  # read value as string
            s = [float(_) for _ in s.split(' ')]  # convert to list of floats
            s = np.log(s) - self.std_viscosity ** 2 / 2  # compute mean of lognormal
            s = self.random_state.lognormal(s, self.std_viscosity)  # sample new values
            s = ' '.join(map(str, s))  # convert to string
            child.attrib['viscosity'] = s  # load to xml
            break

    def render_off_screen(self):
        self.sample_new_env()
        mujoco_py.MjRenderContextOffscreen(self.env.sim, device_id=0)

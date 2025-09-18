import hashlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from fpbnn.utils import camel_to_snake, keys_for_module, paths

CACHE_DIR = Path(".cache/next_states")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _batch_cache_key(name: str, seed: int | None, states: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(name.encode("utf-8"))
    h.update(np.int64(-1 if seed is None else seed).tobytes())
    h.update(np.ascontiguousarray(states).tobytes())
    return h.hexdigest()


class AbstractRegressionEnv(ABC):
    def __init__(
        self,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
        verbose: int = 1,
    ) -> None:
        self.verbose = verbose
        self.noise_std = noise_std
        self.x_range = x_range
        self.input_dim = 1
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.name = camel_to_snake(self.__class__.__name__)
        self.dtype = np.float32

    def handle(self, x: np.ndarray) -> np.ndarray:
        if np.ndim(x) == 1:
            x = x[..., None]
        return x.astype(self.dtype)

    def sample(self, n_samples: int, batch_size: int, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
        x_samples, y_samples = [], []
        for _ in range(n_samples):
            x = self.random_state.uniform(*self.x_range, [batch_size, self.input_dim]).astype(np.float32)
            y = []
            for _ in range(n_particles):
                f = self.sample_function()
                y.append(f(x))
            y = np.stack(y)
            y += self.random_state.normal(0, self.noise_std, y.shape)
            x_samples.append(x)
            y_samples.append(y)
        return np.stack(x_samples), np.stack(y_samples)

    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        meta_train_tuples = []
        for _ in range(n_tasks):
            f = self.sample_function()
            x = self.random_state.uniform(*self.x_range, [n_samples, self.input_dim])
            y = f(x)
            y += self.random_state.normal(0, self.noise_std, y.shape)
            meta_train_tuples.append((x, y))
        return meta_train_tuples

    def generate_meta_test_data(
        self, n_tasks: int, n_samples_context: int, n_samples_test: int
    ) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
        """
        Args:
            n_tasks: number of runs
            n_samples_context: number of training points
            n_samples_test: number of test points
        """
        assert n_samples_test > 0, f"{n_samples_test=}<=0"
        meta_test_tuples = []
        for _ in range(n_tasks):
            f = self.sample_function()
            x = self.random_state.uniform(*self.x_range, [n_samples_context + n_samples_test, self.input_dim])
            y = f(x)
            y += self.random_state.normal(0, self.noise_std, y.shape)
            x_train, x_test = x[:n_samples_context], x[n_samples_context:]
            y_train, y_test = y[:n_samples_context], y[n_samples_context:]
            meta_test_tuples.append((x_train, y_train, x_test, y_test))
        return meta_test_tuples

    @abstractmethod
    def sample_function(self):
        pass

    def sample_train_test(self, n_train: int, n_test: int):
        """
        :param n_train: number of training points
        :param n_test: number of test points
        :return: two tuples containing observations and labels for train and test
        """
        assert n_train > 0, f"{n_train=}<=0"
        assert n_test > 0, f"{n_test=}<=0"
        f = self.sample_function()

        x_train = self.random_state.uniform(*self.x_range, [n_train, self.input_dim]).astype(np.float32)
        y_train = f(x_train)

        y_train += self.random_state.normal(0, self.noise_std, y_train.shape)

        x_test = self.random_state.uniform(*self.x_range, [n_test, self.input_dim]).astype(np.float32)
        y_test = f(x_test)
        return (x_train, y_train), (x_test, y_test)


class Sinusoids(AbstractRegressionEnv):
    def __init__(
        self,
        amplitude: tuple[float, float] = (0.9, 1.1),
        period: tuple[float, float] = (0.9, 1.1),
        slope: tuple[float, float] = (0.0, 0.01),
        x_shift: tuple[float, float] = (0.0, 0.01),
        y_shift: tuple[float, float] = (0.0, 0.01),
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(x_range=x_range, noise_std=noise_std, seed=seed, verbose=verbose)
        self.amplitude = amplitude
        self.period = period
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.slope = slope

    def sample_function(self) -> Callable[[np.ndarray], np.ndarray]:
        a = self.random_state.uniform(*self.amplitude)
        t = self.random_state.uniform(*self.period)
        m = self.random_state.normal(*self.slope)
        x0 = self.random_state.normal(*self.x_shift)
        y0 = self.random_state.normal(*self.y_shift)

        def g(x: np.ndarray) -> np.ndarray:
            return m * x + a * np.sin(2 * np.pi / t * (x - x0)) + y0

        return g


class Densities(AbstractRegressionEnv):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        mean_range: tuple[float, float] = (-0.5, 0.5),
        variance_range: tuple[float, float] = (0.25, 0.5),
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.05,
        seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(x_range=x_range, noise_std=noise_std, seed=seed, verbose=verbose)
        self.variance_range = variance_range
        self.mean_range = mean_range
        self.input_dim = input_dim
        self.output_dim = output_dim
        if tf is None:
            self.dtype = np.float32
        else:
            self.dtype = tf.float32
        self.random_state = np.random.default_rng(self.seed)

    def handle(self, x: np.ndarray) -> np.ndarray:
        if tf is None:
            x = np.asarray(x)
            if np.ndim(x) == 1:
                x = x[..., None]
            return x.astype(np.float32)
        else:
            if tf.rank(x) == 1:
                x = tf.expand_dims(x, axis=-1)
            return tf.cast(x, self.dtype)

    def sample_function(self, mixtures: int = 1) -> Callable[[np.ndarray], np.ndarray]:
        if tf is None or tfp is None:
            means = self.random_state.uniform(
                low=self.mean_range[0],
                high=self.mean_range[1],
                size=(self.output_dim, mixtures, self.input_dim),
            ).astype(np.float32)
            variances = self.random_state.uniform(
                low=self.variance_range[0],
                high=self.variance_range[1],
                size=(self.output_dim, mixtures, self.input_dim),
            ).astype(np.float32)
            weights = np.full((mixtures,), 1.0 / mixtures, dtype=np.float32)

            def g(x: np.ndarray) -> np.ndarray:
                x = self.handle(x)
                y_components = []
                for i in range(self.output_dim):
                    xm = x[:, None, :] - means[i]
                    var = variances[i]
                    log_prob = -0.5 * np.sum((xm * xm) / var, axis=-1) - 0.5 * np.sum(
                        np.log(2 * np.pi * var),
                        axis=-1,
                    )
                    comp = np.exp(log_prob) * weights[None, :]
                    y_components.append(np.sum(comp, axis=-1))
                return np.stack(y_components, axis=-1).astype(np.float32)

            return g

        tfd = tfp.distributions
        mean = self.random_state.uniform(
            low=self.mean_range[0],
            high=self.mean_range[1],
            size=(self.output_dim, mixtures, self.input_dim),
        ).astype(np.float32)
        variance = self.random_state.uniform(
            low=self.variance_range[0],
            high=self.variance_range[1],
            size=(self.output_dim, mixtures, self.input_dim),
        ).astype(np.float32)
        probs = [1 / mixtures for _ in range(mixtures)]
        mixture_distribution = tfd.Categorical(probs=probs)

        def g(x: np.ndarray) -> np.ndarray:
            x = self.handle(x)
            y = []
            for i in range(self.output_dim):
                dist = tfd.Independent(tfd.Normal(mean[i], variance[i]), 1)
                mixture = tfd.MixtureSameFamily(mixture_distribution, dist)
                y.append(mixture.log_prob(x))
            return tf.exp(tf.stack(y, -1))

        return g


class AbstractMujocoEnv(AbstractRegressionEnv, ABC):
    def __init__(
        self,
        env: Any,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(x_range=x_range, noise_std=noise_std, seed=seed, verbose=verbose)

        self.std_capsule = None
        self.env = env

        self.state_space_size = self.env.model.nq + self.env.model.nv
        self.action_space_size = np.prod(self.env.action_space.shape)

        self.qpos_size = self.env.model.nq
        self.qvel_size = self.env.model.nv

        self.input_dim = self.state_space_size + self.action_space_size
        self.output_dim = self.state_space_size

        self.state_range = np.stack([self.env.observation_space.low, self.env.observation_space.high]).tolist()
        self.action_range = np.stack([self.env.action_space.low, self.env.action_space.high]).tolist()

        self.env.reset(seed=self.seed)

        from importlib.resources import files

        self._original_xml_path = files("gymnasium.envs.mujoco.assets") / f"{self.name}.xml"

        self._base_spec: Any | None = None

    def sample_function(self) -> Callable[[np.ndarray], np.ndarray]:
        def g(states: np.ndarray) -> np.ndarray:
            """
            states: (m, d + a)
            next_states: (m, d)
            """
            key = _batch_cache_key(self.name, self.seed, states)
            npz = CACHE_DIR / f"{key}.npz"
            if npz.exists():
                return np.load(npz)["next_states"]

            next_states = []
            disable = self.verbose < 1 or len(states) < 1000
            for state in tqdm(states, desc="generating next states", disable=disable, leave=False):
                next_states.append(self.step(state))
            next_states = np.stack(next_states)

            np.savez_compressed(npz, next_states=next_states)
            return next_states

        return g

    def step(self, state: np.ndarray) -> np.ndarray:
        self.sample_new_env()
        state, action = np.split(state, [self.state_space_size])
        qpos, qvel = np.split(state, [self.qpos_size])
        self.env.data.qpos[:] = qpos
        self.env.data.qvel[:] = qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.step(action)
        self.env.close()
        return np.concatenate([self.env.data.qpos.flat, self.env.data.qvel.flat]).astype(np.float32)

    def run_random_steps(self, n_steps: int):
        for _ in range(n_steps):
            action = self.env.action_space.sample()
            self.env.step(action)

    def sample_new_env(self):
        """Create a new environment with randomized physics parameters using mjSpec."""
        if self._base_spec is None:
            xml_content = self._original_xml_path.read_text(encoding="utf-8")
            mjspec_class = getattr(mujoco, "MjSpec", None)
            if mjspec_class is not None:
                self._base_spec = mjspec_class.from_string(xml_content)
            else:
                self._base_spec = type("MockMjSpec", (), {"from_string": lambda x: None, "compile": lambda: None})()

        spec = self._base_spec.copy()

        self.modify_spec(spec)

        model = spec.compile()
        self.env.model = model
        self.env.data = mujoco.MjData(model)
        self.env.frame_skip = 1

    @abstractmethod
    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        """
        Modify the MjSpec with randomized physics parameters.
        Override this method in child classes to apply specific modifications.

        Args:
            spec: The MjSpec object to modify
        """
        pass

    def modify_capsule_geometries(self, spec: mujoco.MjSpec) -> None:
        """Modify capsule geometry properties using mjSpec API."""
        assert self.std_capsule is not None

        for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
            if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                original_sizes = np.array(geom.size)

                if original_sizes[0] <= 0:
                    continue  # Skip if radius is zero or negative

                if len(original_sizes) > 1 and original_sizes[1] > 0:
                    meaningful_dims = 2  # radius and half-length
                else:
                    meaningful_dims = 1  # radius only

                relevant_sizes = original_sizes[:meaningful_dims]

                mu = np.log(relevant_sizes) - self.std_capsule**2 / 2
                sigma = self.std_capsule
                sampled_sizes = self.random_state.lognormal(mu, sigma)

                min_scale = 0.5
                max_scale = 2.0
                clamped_sizes = np.clip(
                    sampled_sizes,
                    a_min=min_scale * relevant_sizes,
                    a_max=max_scale * relevant_sizes,
                )

                if meaningful_dims >= 2:
                    clamped_sizes[0] = max(clamped_sizes[0], 0.02)  # radius
                    clamped_sizes[1] = max(clamped_sizes[1], 0.05)  # half-length
                else:
                    clamped_sizes[0] = max(clamped_sizes[0], 0.05)  # radius

                new_sizes = original_sizes.copy()
                new_sizes[:meaningful_dims] = clamped_sizes
                geom.size = new_sizes

    def modify_joint_stiffness(self, spec: mujoco.MjSpec) -> None:
        """Modify joint stiffness properties using mjSpec API."""
        assert self.std_stiffness is not None

        for joint in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_JOINT):
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                if joint.stiffness is not None and joint.stiffness > 0:
                    original_stiffness = joint.stiffness

                    mu = np.log(original_stiffness) - self.std_stiffness**2 / 2
                    sampled = self.random_state.lognormal(mu, self.std_stiffness)

                    min_scale = 0.3
                    max_scale = 3.0
                    clamped = np.clip(
                        sampled,
                        a_min=min_scale * original_stiffness,
                        a_max=max_scale * original_stiffness,
                    )

                    joint.stiffness = float(clamped)

    def modify_viscosity(self, spec: mujoco.MjSpec) -> None:
        """Modify viscosity properties using mjSpec API."""
        assert self.std_viscosity is not None

        option = spec.option
        if option.viscosity is not None and option.viscosity > 0:
            original_viscosity = option.viscosity

            mu = np.log(original_viscosity) - self.std_viscosity**2 / 2
            sampled = self.random_state.lognormal(mu, self.std_viscosity)

            option.viscosity = float(sampled)

    def remove_floor(self, spec: mujoco.MjSpec) -> None:
        """
        Remove all floor/ground planes and background elements for deepest black background.

        This method removes all plane geometries and other potential background elements
        to achieve the same deep black void background as the ant environment.
        """
        planes_to_remove = []
        for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                planes_to_remove.append(geom)
            elif geom.name and geom.name.lower() in ["floor", "ground", "wall", "background", "plane"]:
                planes_to_remove.append(geom)

        for plane_geom in planes_to_remove:
            spec.delete(plane_geom)

    def remove_background_elements(self, spec: mujoco.MjSpec) -> None:
        """
        Remove additional background visual elements for deeper black background.

        This removes lights, materials, and other elements that might create
        visual artifacts in the background.
        """
        if hasattr(spec, "worldbody"):
            lights_to_remove = []
            for light in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_LIGHT):
                if light.name and "ambient" in light.name.lower():
                    lights_to_remove.append(light)

            for light in lights_to_remove:
                spec.delete(light)

    def add_black_skybox(self, spec: mujoco.MjSpec) -> None:
        """
        Configure visual settings to ensure the deepest black background like ant.

        This method configures visual properties to minimize atmospheric effects
        and ensures the deepest black void background. The actual black background
        is set in the rendering methods using proper MuJoCo APIs.
        """
        if not hasattr(spec, "visual") or spec.visual is None:
            spec.visual = spec.add_visual()
        if not hasattr(spec.visual, "global_") or spec.visual.global_ is None:
            spec.visual.global_ = spec.visual.add_global()

        if not hasattr(spec.visual, "map") or spec.visual.map is None:
            spec.visual.map = spec.visual.add_map()

        visual_global = spec.visual.global_

        if hasattr(visual_global, "fovy"):
            pass

        self.remove_background_elements(spec)

        pass

    def _configure_clean_rendering(self, renderer: mujoco.Renderer) -> None:
        """Configure MuJoCo scene for a solid background and no artefacts."""

        scene = renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_ADDITIVE] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        if hasattr(renderer, "model") and hasattr(renderer.model.vis, "quality"):
            assert hasattr(renderer.model.vis.quality, "offsamples"), (
                "Renderer quality should have offsamples attribute"
            )
            renderer.model.vis.quality.offsamples = 0  # offscreen multisample count

    def _render_text_with_monospace_numbers(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        position: tuple[int, int],
        regular_font: ImageFont.ImageFont,
        monospace_font: ImageFont.ImageFont,
        fill: str = "white",
    ) -> tuple[int, int]:
        """
        Render text with monospace font for numbers and regular font for text.
        Returns the dimensions (width, height) of the rendered text.
        """
        import re

        parts = re.split(r"(\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)

        x, y = position
        current_x = x
        max_height = 0

        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                continue

            is_numeric = re.match(r"\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?", part)
            font = monospace_font if is_numeric else regular_font

            bbox = draw.textbbox((0, 0), part, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            max_height = max(max_height, text_height)

            draw.text((current_x, y), part, fill=fill, font=font)
            current_x += text_width

        return current_x - x, max_height

    def create_state_comparison_image(
        self,
        predicted_state: np.ndarray,
        predicted_uncertainty: np.ndarray,
        ground_truth_state: np.ndarray,
        output_path: str | None = None,
        resolution: tuple[int, int] = (1500, 1000),
        use_fixed_env: bool = False,
        fixed_model=None,
        fixed_data=None,
        iteration: int | None = None,
        total_iterations: int | None = None,
        nll: float | None = None,
        rmse: float | None = None,
    ) -> Path:
        """
        Create an elegant overlapped visualization showing:
        All overlapped in a single frame for direct comparison.

        Args:
            predicted_state: Model's predicted next state
            predicted_uncertainty: Standard deviation of prediction (used for opacity)
            ground_truth_state: Actual next state
            output_path: Optional custom output path
            resolution: (width, height) for the image

        Returns:
            Path to the created comparison image
        """
        import numpy as np
        from PIL import Image, ImageDraw

        width, height = resolution

        if use_fixed_env and fixed_model is not None and fixed_data is not None:
            current_model = fixed_model
            current_data = fixed_data
        else:
            self.sample_new_env()
            current_model = self.env.model
            current_data = self.env.data

        if hasattr(current_model.vis, "global_"):
            current_model.vis.global_.offwidth = int(width)
            current_model.vis.global_.offheight = int(height)
        elif hasattr(current_model.vis, "global"):
            vis_global = getattr(current_model.vis, "global")
            vis_global.offwidth = int(width)
            vis_global.offheight = int(height)

        original_rgba = current_model.geom_rgba.copy()

        renderer = mujoco.Renderer(current_model, height=height, width=width)

        self._configure_clean_rendering(renderer)

        if self.name == "ant":
            qpos_size = current_model.nq
            gt_qpos = ground_truth_state[:qpos_size]
            gt_qvel = ground_truth_state[qpos_size:]

            original_qpos = current_data.qpos.copy()
            original_qvel = current_data.qvel.copy()

            current_data.qpos[:] = gt_qpos
            current_data.qvel[:] = gt_qvel
            mujoco.mj_forward(current_model, current_data)

            camera = self._select_camera(current_model, current_data)

            current_data.qpos[:] = original_qpos
            current_data.qvel[:] = original_qvel
            mujoco.mj_forward(current_model, current_data)
        else:
            camera = self._select_camera(current_model, current_data)

        states_to_render = [
            (ground_truth_state, [0.3, 1.0, 0.3], 0.9, "solid"),  # Green, more visible
            (predicted_state, [1.0, 0.3, 0.3], None, "solid"),  # Red, uncertainty-based opacity
        ]

        rendered_frames = []

        for state, color, base_opacity, render_mode in states_to_render:
            qpos_size = current_model.nq
            qpos = state[:qpos_size]
            qvel = state[qpos_size:]

            current_data.qpos[:] = qpos
            current_data.qvel[:] = qvel
            mujoco.mj_forward(current_model, current_data)

            if base_opacity is None:  # This is the predicted state
                mean_uncertainty = np.mean(predicted_uncertainty)
                max_uncertainty = np.max(predicted_uncertainty) if np.max(predicted_uncertainty) > 0 else 1.0
                uncertainty_ratio = mean_uncertainty / max_uncertainty
                opacity = max(0.6, 0.95 - uncertainty_ratio * 0.35)
            else:
                opacity = base_opacity

            if render_mode == "wireframe":
                self._set_material_properties(current_model, opacity * 0.5, color)
            else:
                self._set_material_properties(current_model, opacity, color)

            renderer.update_scene(current_data, camera=camera)
            frame = renderer.render()
            rendered_frames.append((frame, opacity))

        if len(rendered_frames) == 2:
            ground_truth_frame, gt_opacity = rendered_frames[0]
            predicted_frame, pred_opacity = rendered_frames[1]

            gt_float = ground_truth_frame.astype(np.float32) / 255.0
            pred_float = predicted_frame.astype(np.float32) / 255.0

            threshold = 0.05
            gt_mask = np.any(gt_float > threshold, axis=2, keepdims=True)
            pred_mask = np.any(pred_float > threshold, axis=2, keepdims=True)

            composite = np.zeros_like(gt_float)

            gt_alpha = gt_mask * gt_opacity
            composite = gt_float * gt_alpha + composite * (1 - gt_alpha)

            pred_alpha = pred_mask * pred_opacity
            composite = pred_float * pred_alpha + composite * (1 - pred_alpha)

            frame = (composite * 255).astype(np.uint8)
        else:
            frame = rendered_frames[-1][0]  # Fallback to last frame

        renderer.close()
        if not use_fixed_env:
            self.env.close()

        current_model.geom_rgba[:] = original_rgba

        combined_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(combined_image)

        import sys
        from pathlib import Path

        from PIL import ImageFont

        font_size = 48 if sys.platform == "darwin" else 36

        def load_system_font(is_monospace: bool, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
            if sys.platform == "darwin":  # macOS
                if is_monospace:
                    font_paths = [
                        "/System/Library/Fonts/Menlo.ttc",
                        "/System/Library/Fonts/Monaco.ttf",
                    ]
                else:
                    font_paths = [
                        "/System/Library/Fonts/Helvetica.ttc",
                        "/System/Library/Fonts/HelveticaNeue.ttc",
                    ]

                for path in font_paths:
                    if Path(path).exists():
                        font = ImageFont.truetype(path, size)
                        assert font is not None, f"Failed to load font from {path}"
                        return font
            else:
                font_name = "DejaVuSansMono.ttf" if is_monospace else "DejaVuSans.ttf"
                font_paths = [
                    font_name,
                    f"/usr/share/fonts/truetype/dejavu/{font_name}",
                    f"/usr/local/share/fonts/{font_name}",
                    f"C:/Windows/Fonts/{font_name}",
                ]
                for path in font_paths:
                    if Path(path).exists():
                        font = ImageFont.truetype(path, size)
                        assert font is not None, f"Failed to load font from {path}"
                        return font

            return ImageFont.load_default()

        mono_font_size = int(font_size * 0.78)  # Reduce mono font by 22%
        text_font = load_system_font(False, font_size)
        mono_font = load_system_font(True, mono_font_size)

        if nll is not None and rmse is not None:
            combined_text = f"iteration {iteration:7d}/{total_iterations}  |  nll: {nll:+.2e}  |  rmse: {rmse:.2e}"
        else:
            combined_text = f"iteration {iteration:7d}/{total_iterations}"

        text_width, text_height = self._render_text_with_monospace_numbers(
            draw,
            combined_text,
            (0, -1000),
            text_font,
            mono_font,
            "white",  # Off-screen to measure
        )
        self._render_text_with_monospace_numbers(
            draw, combined_text, ((width - text_width) // 2, 15), text_font, mono_font, "white"
        )

        if output_path is None:
            output_dir = paths.state_comparisons
            output_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = output_dir / f"{self.name}_overlapped_comparison.png"
        else:
            final_output_path = Path(output_path)

        combined_image.save(final_output_path)
        return final_output_path

    def _set_material_properties(self, model, opacity: float, color: list[float]):
        """Set material properties for uncertainty visualization."""
        for i in range(model.ngeom):
            model.geom_rgba[i, 3] = opacity
            model.geom_rgba[i, :3] = color

    def create_gif_from_model(
        self,
        output_path: str,
        model_predict_func: Callable[[np.ndarray], np.ndarray],
        n_episodes: int = 3,
        episode_length: int = 100,
        fps: int = 10,
        resolution: tuple[int, int] = (1500, 1000),
    ) -> Path:
        """
        Create a GIF showing the environment running with predictions from a trained model.
        """
        import imageio

        frames = []
        width, height = resolution

        original_random_state = self.random_state.get_state()
        self.random_state.set_state(np.random.RandomState(self.seed).get_state())
        self.sample_new_env()
        self.random_state.set_state(original_random_state)

        total_frames = int(n_episodes) * int(episode_length)
        with tqdm(total=total_frames, desc="Creating GIF", unit="frame") as pbar:
            for episode in range(n_episodes):
                self.env.reset(seed=self.seed)

                if hasattr(self.env.model.vis, "global_"):
                    self.env.model.vis.global_.offwidth = int(width)
                    self.env.model.vis.global_.offheight = int(height)
                elif hasattr(self.env.model.vis, "global"):
                    vis_global = getattr(self.env.model.vis, "global")
                    vis_global.offwidth = int(width)
                    vis_global.offheight = int(height)

                renderer = mujoco.Renderer(self.env.model, height=height, width=width)

                self._configure_clean_rendering(renderer)

                if hasattr(self, "std_stiffness") and hasattr(self, "std_capsule"):
                    ground_truth_env = self.__class__(
                        std_stiffness=self.std_stiffness,
                        std_capsule=self.std_capsule,
                        x_range=self.x_range,
                        noise_std=self.noise_std,
                        seed=self.seed,
                    )
                else:
                    ground_truth_env = self.__class__(x_range=self.x_range, noise_std=self.noise_std, seed=self.seed)
                ground_truth_env.env.reset(seed=self.seed)

                if self.name == "ant":
                    fixed_camera = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(fixed_camera)
                    extent = float(ground_truth_env.model.stat.extent)
                    fixed_camera.lookat[:] = np.array([0.0, 0.0, 0.0])
                    fixed_camera.azimuth = 0.0
                    fixed_camera.elevation = -90.0
                    fixed_camera.distance = 4.5 * extent  # Reduced from 6.0 to zoom in more
                else:
                    fixed_camera = None

                for step in range(episode_length):
                    current_state = np.concatenate([self.env.data.qpos.flat, self.env.data.qvel.flat]).astype(
                        np.float32
                    )
                    action = self.env.action_space.sample()

                    ground_truth_env.step(action)
                    self.env.step(action)

                    model_input = np.concatenate([current_state, action]).reshape(1, -1)
                    predicted_next_state = model_predict_func(model_input)
                    if hasattr(predicted_next_state, "numpy"):
                        predicted_next_state = predicted_next_state.numpy()
                    predicted_next_state = predicted_next_state.flatten()
                    assert len(predicted_next_state) == self.state_space_size, (
                        f"Expected {self.state_space_size} state dims"
                    )

                    qpos_size = self.env.model.nq
                    predicted_qpos = predicted_next_state[:qpos_size]
                    predicted_qvel = predicted_next_state[qpos_size:]

                    assert len(predicted_qpos) == len(self.env.data.qpos), "qpos dimension mismatch"
                    assert len(predicted_qvel) == len(self.env.data.qvel), "qvel dimension mismatch"
                    self.env.data.qpos[:] = predicted_qpos
                    self.env.data.qvel[:] = predicted_qvel
                    mujoco.mj_forward(self.env.model, self.env.data)

                    if self.name == "ant" and fixed_camera is not None:
                        camera = fixed_camera
                    else:
                        camera = self._select_camera(self.env.model, ground_truth_env.data)

                    renderer.update_scene(self.env.data, camera=camera)
                    frame = renderer.render()
                    pil_image = Image.fromarray(frame)
                    frames.append(np.array(pil_image))
                    pbar.update(1)

                renderer.close()
                ground_truth_env.close()
        self.env.close()

        assert output_path is not None, "output_path is required"
        final_output_path = Path(output_path)

        if frames:
            frame_duration = 1.0 / float(fps)
            imageio.mimsave(final_output_path, frames, duration=frame_duration, loop=0, optimize=True)

        return final_output_path

    def _select_camera(self, model: mujoco.MjModel, data: mujoco.MjData | None = None):
        """
        Return a suitable camera or camera id for rendering.
        """
        preferred_camera_names: list[str] = []
        if self.name in ["half_cheetah", "swimmer", "hopper"]:
            preferred_camera_names.append("track")
        if self.name == "hopper":
            preferred_camera_names.append("trackcom")

        for camera_name in preferred_camera_names:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id != -1:
                return cam_id

        if self.name == "inverted_double_pendulum" and data is not None:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = np.array([0.0, 0.0, 1.0])
            cam.azimuth = 90.0
            cam.elevation = -15.0
            cam.distance = 3.5  # Reduced from 5.0 to zoom in more
            return cam

        if self.name == "hopper" and data is not None:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            if torso_id != -1:
                cam.trackbodyid = torso_id
                cam.lookat[:] = np.array(data.xpos[torso_id])
            else:
                cam.lookat[:] = np.array(model.stat.center)
            cam.azimuth = 90.0
            cam.elevation = -20.0
            cam.distance = 2.0 * float(model.stat.extent)
            return cam

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)

        center = np.array(model.stat.center)
        extent = float(model.stat.extent)
        cam.lookat[:] = center

        if self.name == "inverted_double_pendulum":
            cam.azimuth = 90.0
            cam.elevation = -15.0
            cam.distance = 3.5  # Reduced from 5.0 to zoom in more
            cam.lookat[:] = np.array([0.0, 0.0, 1.0])
        elif self.name == "inverted_pendulum":
            cam.azimuth = 90.0
            cam.elevation = -10.0
            cam.distance = 1.5 * extent  # Reduced from 2.0 to zoom in more
        elif self.name == "hopper":
            cam.azimuth = 90.0
            cam.elevation = -20.0
            cam.distance = 0.75 * extent
            cam.lookat[2] = 1.15
        elif self.name == "reacher":
            cam.azimuth = 0.0
            cam.elevation = -90.0
            cam.distance = 1.5 * extent
        elif self.name == "ant" and data is not None:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            if torso_id != -1:
                cam.lookat[:] = np.array(data.xpos[torso_id])
            else:
                cam.lookat[:] = center
            cam.azimuth = 0.0
            cam.elevation = -90.0
            cam.distance = 3.0 * extent  # Reduced from 4.0 to zoom in more
            return cam
        elif self.name == "ant":
            cam.azimuth = 0.0
            cam.elevation = -90.0
            cam.distance = 3.0 * extent  # Reduced from 4.0 to zoom in more
        elif self.name == "half_cheetah":
            cam.azimuth = 90.0
            cam.elevation = -15.0
            cam.distance = 4.0 * extent
        elif self.name == "swimmer":
            cam.azimuth = 0.0
            cam.elevation = -90.0
            cam.distance = 4.0 * extent
        else:
            cam.azimuth = 60.0
            cam.elevation = -20.0
            cam.distance = 3.0 * extent

        return cam


class Ant(AbstractMujocoEnv):
    def __init__(
        self,
        std_capsule: float = 0.3,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.ant_v5 import AntEnv as _Env

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_capsule = std_capsule

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_capsule_geometries(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


class HalfCheetah(AbstractMujocoEnv):
    """
    Randomized Half Cheetah environment
    """

    def __init__(
        self,
        std_stiffness: float = 0.1,
        std_capsule: float = 0.1,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv as _Env

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_stiffness = std_stiffness
        self.std_capsule = std_capsule

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_capsule_geometries(spec)
        self.modify_joint_stiffness(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


class Hopper(AbstractMujocoEnv):
    def __init__(
        self,
        std_capsule: float = 0.15,
        std_stiffness: float = 0.1,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.hopper_v5 import HopperEnv as _Env

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_capsule = std_capsule
        self.std_stiffness = std_stiffness

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_capsule_geometries(spec)
        self.modify_joint_stiffness(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


class InvertedPendulum(AbstractMujocoEnv):
    def __init__(
        self,
        std_capsule: float = 0.1,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.inverted_pendulum_v5 import (
            InvertedPendulumEnv as _Env,
        )

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_capsule = std_capsule

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_capsule_geometries(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


class InvertedDoublePendulum(AbstractMujocoEnv):
    def __init__(
        self,
        std_capsule: float = 0.1,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import (
            InvertedDoublePendulumEnv as _Env,
        )

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_capsule = std_capsule

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_capsule_geometries(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


class Reacher(AbstractMujocoEnv):
    def __init__(
        self,
        std_stiffness: float = 0.1,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv as _Env

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_stiffness = std_stiffness

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_joint_stiffness(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


class Swimmer(AbstractMujocoEnv):
    def __init__(
        self,
        std_capsule: float = 0.1,
        std_viscosity: float = 0.1,
        x_range: tuple[float, float] = (-1, 1),
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv as _Env

        super().__init__(
            _Env(render_mode=None),
            x_range=x_range,
            noise_std=noise_std,
            seed=seed,
        )
        self.std_capsule = std_capsule
        self.std_viscosity = std_viscosity

    def modify_spec(self, spec: mujoco.MjSpec) -> None:
        self.modify_capsule_geometries(spec)
        self.modify_viscosity(spec)
        self.remove_floor(spec)
        self.add_black_skybox(spec)


available_envs = keys_for_module(sys.modules[__name__], "snake")

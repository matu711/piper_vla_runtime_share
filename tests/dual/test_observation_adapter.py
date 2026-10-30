"""Runtime observation contract tests; no Bundle loading or devices."""
import unittest
import numpy as np
from piper_vla_dual_runtime.observation_adapter import adapt_runtime_observation, TRAINING_CAMERAS
from piper_vla_dual_runtime.policy_transforms import IMAGE_KEYS, STATE_KEY

class AdapterTests(unittest.TestCase):

    def observation(self):
        return {STATE_KEY: np.arange(14, dtype=np.float32) / 100, 'prompt': 'Assembling a box.', **{key: ((np.arange(7 * 9 * 3).reshape(7, 9, 3) + 31 * index) % 256).astype(np.uint8) for (index, key) in enumerate(IMAGE_KEYS.values())}}

    def test_preserve_state_prompt_cameras_rgb_and_values(self):
        original = self.observation()
        adapted = adapt_runtime_observation(original)
        np.testing.assert_array_equal(adapted['state'], original[STATE_KEY])
        self.assertEqual(adapted['state'].dtype, np.float32)
        self.assertEqual(adapted['prompt'], original['prompt'])
        for (camera, training_key) in TRAINING_CAMERAS.items():
            result = adapted['images'][training_key]
            self.assertEqual(result.shape, (3, 7, 9))
            self.assertEqual(result.dtype, np.uint8)
            np.testing.assert_array_equal(result.transpose(1, 2, 0), original[IMAGE_KEYS[camera]])
        adapted['state'][0] = 99
        self.assertEqual(original[STATE_KEY][0], 0)

    def test_reject_invalid_shape_type_or_double_adaptation(self):
        for (key, value) in ((STATE_KEY, np.zeros(7)), (STATE_KEY, np.full(14, np.nan)), (IMAGE_KEYS['front'], np.zeros((3, 7, 9), np.uint8)), (IMAGE_KEYS['front'], np.zeros((7, 9, 3), np.float32))):
            observation = self.observation()
            observation[key] = value
            with self.assertRaises(ValueError):
                adapt_runtime_observation(observation)
        with self.assertRaises(ValueError):
            adapt_runtime_observation(adapt_runtime_observation(self.observation()))

"""CPU-only compatibility checks: no ROS, sockets, model imports or weights."""
import ast
import dataclasses
from pathlib import Path
import types
import socket
import threading
import unittest
from unittest import mock

import cv2
import numpy as np
from openpi_client import msgpack_numpy

from piper_vla_common.adapters import prepare_image
from piper_vla_common.execution import ActionContract, ActionContractError, ChunkQueue, Session
from piper_vla_common.observations import ObservationStore, decode_rgb
from piper_vla_common.policy_client import (OpenPIClient, InferenceRequest,
    PolicyCommunicationError, PolicyTimeout, StaleResponse)
from piper_vla_runtime.observation_adapter import policy_state
from piper_vla_common.model_contracts import validate_normalization_config

from piper_vla_dual_runtime.core import vector
from piper_vla_dual_runtime.policy_transforms import STATE_KEY, IMAGE_KEYS, MODEL_IMAGES

ROOT = Path(__file__).resolve().parents[2]


def numeric_transforms():
    source = ROOT / "vendor/openpi_pbl5/src/openpi/transforms.py"
    names = {"Normalize", "Unnormalize", "DeltaActions", "AbsoluteActions", "PadStatesAndActions",
             "TokenizePrompt", "apply_tree", "pad_to_dim", "_assert_quantile_stats"}
    selected = [node for node in ast.parse(source.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    scope = dict(np=np, dataclasses=dataclasses, DataTransformFn=object,
                 flatten_dict=lambda data: dict(data), unflatten_dict=lambda data: dict(data))
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"), scope)
    return scope


class DualInputs:
    def __call__(self, data):
        images = {}
        for name, key in IMAGE_KEYS.items():
            image = np.asarray(data[key])
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError(key + " must be uint8[224,224,3]")
            images[MODEL_IMAGES[name]] = image
        result = {"state": vector(data[STATE_KEY], 14).astype(np.float32),
                  "image": images, "image_mask": {key: np.True_ for key in images},
                  "prompt": data["prompt"]}
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError("training actions must be finite [50,14]")
            result["actions"] = actions.copy()
        return result


class DualOutputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.shape != (50, 32) or not np.isfinite(actions).all():
            raise ValueError("model actions must be finite [50,32]")
        return {"actions": actions[:, :14].copy()}


class ObservationCompatibilityTests(unittest.TestCase):
    def test_mixed_or_unknown_normalization_is_rejected_without_conversion(self):
        for mode in (True, False):
            validate_normalization_config(types.SimpleNamespace(use_quantile_norm=mode))
        for data in (types.SimpleNamespace(use_quantile_norm="quantile"),
                     types.SimpleNamespace(use_quantile_norm=True, standard_norm_keys=["state"])):
            with self.assertRaises(ValueError):
                validate_normalization_config(data)

    def test_snapshots_preserve_prompt_generation_stamps_and_post_start_gate(self):
        store = ObservationStore(("state", "front", "wrist", "ee"))
        baseline = dict(store.versions)
        for name in store.names:
            store.update(name, types.SimpleNamespace(header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(to_sec=lambda: 123.))), received=10.)
        first = store.capture(3, "first", baseline)
        self.assertEqual(first.generation, 3)
        self.assertEqual(first.source_stamps["ee"], 123.)
        self.assertEqual(first.received["ee"], 10.)
        with self.assertRaisesRegex(ValueError, "post_start"):
            store.capture(4, "second", dict(store.versions))
        store.update("state", object(), 11.)
        self.assertEqual(first.versions["state"], 1)
        self.assertEqual(first.prompt, "first")
        with self.assertRaises(TypeError):
            first.messages["state"] = object()

    def test_rgb_rotation_resize_and_pbl8_states_remain_identical(self):
        bgr = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
        ok, data = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        rgb = decode_rgb(types.SimpleNamespace(data=data.tobytes()))
        np.testing.assert_equal(rgb, bgr[:, :, ::-1])
        from openpi_client import image_tools
        expected = image_tools.resize_with_pad(cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE), 224, 224)
        np.testing.assert_equal(prepare_image(rgb, rotation=1), expected)
        msg = types.SimpleNamespace(position=[.1] * 6 + [.05])
        cfg = dict(gripper_open_width_m=.1, finger_qpos_max_m=.035)
        seven = policy_state(msg, {"observation": {"state": {"shape": [7]}}}, True, cfg)
        np.testing.assert_equal(seven, np.asarray(msg.position, np.float32))
        eight = policy_state(msg, {}, True, cfg)
        np.testing.assert_allclose(eight, [.1] * 6 + [.0175, .0175])

    def test_libero_images_and_masks_for_both_model_families(self):
        source = ROOT / "vendor/openpi_pbl5/src/openpi/policies/libero_policy.py"
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ("LiberoInputs", "LiberoOutputs", "_parse_image")]
        family = types.SimpleNamespace(PI0="pi0", PI05="pi05", PI0_FAST="fast")
        scope = dict(np=np, dataclasses=dataclasses, transforms=types.SimpleNamespace(DataTransformFn=object),
                     _model=types.SimpleNamespace(ModelType=family))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
        obs = {"observation/state": np.arange(7, dtype=np.float32),
               "observation/image": np.full((224, 224, 3), 71, np.uint8),
               "observation/wrist_image": np.full((224, 224, 3), 137, np.uint8), "prompt": "test"}
        for model in (family.PI0, family.PI05):
            result = scope["LiberoInputs"](model)(obs)
            np.testing.assert_equal(result["state"], obs["observation/state"])
            np.testing.assert_equal(result["image"]["base_0_rgb"], obs["observation/image"])
            np.testing.assert_equal(result["image"]["left_wrist_0_rgb"], obs["observation/wrist_image"])
            self.assertFalse(result["image_mask"]["right_wrist_0_rgb"])
            self.assertFalse(result["image"]["right_wrist_0_rgb"].any())

    def test_standard_and_quantile_joint_roundtrips_use_request_state_once(self):
        t = numeric_transforms()
        for width, quantile in ((7, False), (7, True), (14, True)):
            mask = (True,) * 6 + (False,)
            if width == 14:
                mask *= 2
            state = np.arange(width) / 20
            absolute = np.tile(state + .03, (50, 1))
            stats = {key: types.SimpleNamespace(mean=np.arange(width) / 10, std=np.full(width, .4),
                     q01=np.full(width, -.7), q99=np.full(width, 1.2)) for key in ("state", "actions")}
            data = t["DeltaActions"](mask)({"state": state.copy(), "actions": absolute.copy()})
            data = t["Normalize"](stats, use_quantiles=quantile)(data)
            data = t["PadStatesAndActions"](32)(data)
            data = t["Unnormalize"](stats, use_quantiles=quantile)(data)
            data = t["AbsoluteActions"](mask)(data)
            np.testing.assert_allclose(data["actions"][:, :width], absolute, atol=1e-6)


class TransportTests(unittest.TestCase):
    def test_explicit_initial_deadline_overrides_only_that_request(self):
        clock = mock.Mock(return_value=100.)
        connection = mock.Mock()
        def receive(*, timeout):
            if connection.recv.call_count == 1:
                return msgpack_numpy.packb({})
            clock.return_value += 3.  # Initial compilation exceeds normal 2s.
            return msgpack_numpy.packb({"actions": [[0.] * 7]})
        connection.recv.side_effect = receive
        client = self.client(connection, clock=clock, timeout=2.)
        client.infer({}, request=InferenceRequest(0, 130., lambda _: True, clock))
        self.assertEqual(client.connector.call_args.kwargs["open_timeout"], 30.)
        self.assertEqual(connection.recv.call_args.kwargs["timeout"], 30.)
        self.assertEqual(client.timeout, 2.)
        with self.assertRaises(PolicyTimeout):
            client.infer({})  # Same connection, next response takes >2s.
        self.assertEqual(connection.recv.call_args.kwargs["timeout"], 2.)
        self.assertIsNone(client.ws)

    def test_send_watchdog_interrupts_only_the_owned_connection(self):
        released = threading.Event()
        connection = mock.Mock()
        connection.socket = mock.Mock(spec=socket.socket)
        connection.socket.shutdown.side_effect = lambda *_: released.set()
        connection.send.side_effect = lambda *_: released.wait(1)
        connection.recv.return_value = msgpack_numpy.packb({})
        client = self.client(connection, timeout=.02)
        with self.assertRaises(PolicyTimeout):
            client.infer({})
        self.assertTrue(released.is_set())
        connection.socket.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.assertIsNone(client.ws)

    def test_deadline_is_rechecked_at_chunk_commit_after_validation(self):
        clock = mock.Mock(return_value=0.)
        chunks = ChunkQueue(20, 15, ActionContract(14, 50))
        generation = chunks.start()
        def validate(*args, **kwargs):
            clock.return_value = 3.
            return np.zeros((50, 14))
        chunks.contract = types.SimpleNamespace(validate=validate, horizon=50)
        request = InferenceRequest(generation, 2., chunks.session.accepts, clock)
        with self.assertRaises(PolicyTimeout):
            chunks.stage(np.zeros((50, 14)), generation, request)
        self.assertIsNone(chunks.pending)

    def client(self, connection, clock=None, **kwargs):
        return OpenPIClient("localhost", 8000, connector=mock.Mock(return_value=connection),
                            clock=clock or __import__("time").monotonic, **kwargs)

    def test_timeout_closes_and_next_request_uses_new_connection(self):
        old, new = mock.Mock(), mock.Mock()
        old.recv.side_effect = [msgpack_numpy.packb({}), TimeoutError("late")]
        new.recv.side_effect = [msgpack_numpy.packb({}), msgpack_numpy.packb({"actions": np.ones((2, 7))})]
        client = self.client(old, action_contract=ActionContract(7))
        client.connector.side_effect = [old, new]
        with self.assertRaises(PolicyTimeout):
            client.infer({})
        self.assertIsNone(client.ws)
        old.close.assert_called_once()
        self.assertEqual(client.infer({})["actions"].shape, (2, 7))
        self.assertEqual(client.connector.call_count, 2)
        client.close()

    def test_absolute_deadline_includes_connection_pack_and_validation(self):
        for phase in ("connect", "send", "validation"):
            clock = mock.Mock(return_value=0.)
            connection = mock.Mock()
            connection.recv.side_effect = [msgpack_numpy.packb({}), msgpack_numpy.packb({"actions": [[0.] * 7]})]
            contract = ActionContract(7)
            client = self.client(connection, clock=clock, action_contract=contract)
            def expire(*args, **kwargs):
                clock.return_value = 3.
                return connection
            if phase == "connect":
                client.connector.side_effect = expire
            elif phase == "send":
                connection.send.side_effect = expire
            else:
                original = contract.validate
                wrapped = mock.Mock(side_effect=lambda *a, **k: (expire(), original(*a, **k))[1])
                client.action_contract = types.SimpleNamespace(validate=wrapped)
            with self.assertRaises(PolicyTimeout):
                client.infer({})
            self.assertIsNone(client.ws)

    def test_stop_restart_during_recv_discards_response(self):
        session = Session(enabled=True)
        clock = lambda: 0.
        connection = mock.Mock()
        def response(*args, **kwargs):
            if connection.recv.call_count == 1:
                return msgpack_numpy.packb({})
            session.advance(False)
            session.advance(True)
            return msgpack_numpy.packb({"actions": [[0.] * 7]})
        connection.recv.side_effect = response
        client = self.client(connection, clock=clock)
        with self.assertRaises(StaleResponse):
            client.infer({}, request=InferenceRequest(0, 2., session.accepts, clock))
        self.assertIsNone(client.ws)

    def test_bad_action_contract_is_not_a_transport_failure(self):
        connection = mock.Mock()
        connection.recv.side_effect = [msgpack_numpy.packb({}), msgpack_numpy.packb({"actions": [[np.nan] * 14]})]
        client = self.client(connection, action_contract=ActionContract(14, 50))
        with self.assertRaises(ActionContractError):
            client.infer({})
        connection.close.assert_called_once()

    def test_chunk_profiles_preserve_prefetch_and_no_recovery_after_stop(self):
        for width, horizon, prefetch in ((7, None, 17), (14, 50, 15)):
            chunks = ChunkQueue(20, prefetch, ActionContract(width, horizon))
            generation = chunks.start()
            actions = np.arange(50 * width).reshape(50, width)
            chunks.stage(actions, generation)
            for index in range(20):
                selected = chunks.select()
                self.assertEqual(selected[1], index)
                np.testing.assert_equal(selected[2], actions[index])
                self.assertEqual(selected[3], index == prefetch)
            with self.assertRaisesRegex(ValueError, "underrun"):
                chunks.select()
            chunks.stop()
            chunks.start()
            self.assertFalse(chunks.stage(actions, generation))

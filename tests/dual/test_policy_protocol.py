"""Mock WebSocket transport tests; no sockets, models or ROS devices."""
import unittest
from unittest import mock
import numpy as np
from openpi_client import msgpack_numpy
from piper_vla_dual_runtime.core import POLICY_CONTRACT
from piper_vla_dual_runtime.policy_client import PolicyClient
from piper_vla_dual_runtime.policy_transforms import IMAGE_KEYS, STATE_KEY
from test_common import DualInputs

class ProtocolTests(unittest.TestCase):

    def test_websocket_request_response_and_metadata(self):
        observation = {STATE_KEY: np.arange(14, dtype=np.float32), 'prompt': 'both arms'}
        observation.update({key: np.full((224, 224, 3), i, np.uint8) for (i, key) in enumerate(IMAGE_KEYS.values())})
        actions = np.ones((50, 14))
        transport = mock.Mock()
        transport.recv.side_effect = [msgpack_numpy.packb(POLICY_CONTRACT), msgpack_numpy.packb({'actions': actions})]
        with mock.patch('piper_vla_dual_runtime.policy_client.connect', return_value=transport) as connect:
            client = PolicyClient('127.0.0.1', 8001, timeout=0.5)
            result = client.infer(observation)
            decoded = msgpack_numpy.unpackb(transport.send.call_args.args[0])
            transformed = DualInputs()(decoded)
            self.assertEqual(len(transformed['image']), 3)
            np.testing.assert_equal(transformed['state'], observation[STATE_KEY])
            np.testing.assert_equal(result['actions'], actions)
            connect.assert_called_once()
            self.assertEqual(transport.recv.call_count, 2)
            timeouts = [call.kwargs['timeout'] for call in transport.recv.call_args_list]
            self.assertTrue(0 < timeouts[1] <= timeouts[0] <= 0.5)
            client.close()
            transport.close.assert_called_once()

    def test_websocket_timeout_bad_metadata_and_server_error(self):
        for response in (TimeoutError('no response'), msgpack_numpy.packb({'action_dim': 7}), 'server error'):
            transport = mock.Mock()
            transport.recv.side_effect = [msgpack_numpy.packb(POLICY_CONTRACT), response] if not isinstance(response, bytes) else [response]
            with mock.patch('piper_vla_dual_runtime.policy_client.connect', return_value=transport):
                client = PolicyClient('127.0.0.1', 8001, timeout=0.1)
                with self.assertRaises((TimeoutError, ValueError, RuntimeError)):
                    client.infer({})
                client.close()

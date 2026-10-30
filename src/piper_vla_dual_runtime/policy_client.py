"""Same OpenPI msgpack WebSocket protocol, with bounded receive time."""
from piper_vla_common.execution import ActionContract
from piper_vla_common.policy_client import OpenPIClient
from websockets.sync.client import connect

from piper_vla_dual_runtime.core import validate_contract


class PolicyClient(OpenPIClient):
    def __init__(self, host, port, timeout=2.0, contract=None):
        super().__init__(host, port, timeout, metadata_validator=lambda metadata: validate_contract(metadata, contract),
                         action_contract=ActionContract(14, 50), connector=connect)

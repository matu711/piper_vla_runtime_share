"""OpenPI MessagePack wire protocol with one bounded request per connection.

No action conversion lives here. A timed-out connection is always discarded,
so an untagged late OpenPI response cannot satisfy a subsequent request.
"""
from dataclasses import dataclass
import socket
import threading
import time

from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from piper_vla_common.execution import ActionContractError


class PolicyCommunicationError(RuntimeError):
    pass


class PolicyTimeout(PolicyCommunicationError, TimeoutError):
    pass


class StaleResponse(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceRequest:
    generation: int
    deadline: float
    is_current: object
    clock: object = time.monotonic

    def check(self):
        if not self.is_current(self.generation):
            raise StaleResponse("inference session is no longer current")
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise PolicyTimeout("inference absolute deadline exceeded")
        return remaining


class OpenPIClient:
    def __init__(self, host, port, timeout=2.0, metadata_validator=None,
                 action_contract=None, minimum_actions=1, finite_rows=None,
                 connector=None, clock=time.monotonic):
        import math
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("inference timeout must be finite and positive")
        self.uri = host if host.startswith(("ws://", "wss://")) else "ws://" + host
        self.uri += ":%d" % port
        self.timeout, self.ws = timeout, None
        self.metadata_validator, self.action_contract = metadata_validator, action_contract
        self.minimum_actions, self.finite_rows = minimum_actions, finite_rows
        self.connector, self.clock = connector or connect, clock
        self.last_timing_ms, self.metadata = {}, {}

    def close(self):
        connection, self.ws = self.ws, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass  # Cleanup cannot turn a transport failure into a safety fault.

    def infer(self, observation, *, request=None):
        # An explicit request owns the whole deadline (e.g. initial compile).
        # self.timeout remains the default for subsequent ordinary requests.
        request = request or InferenceRequest(0, self.clock() + self.timeout, lambda _: True, self.clock)
        started = self.clock()
        timer = None
        expiry_lock = threading.Lock()
        completed = False
        try:
            if self.ws is None:
                self.ws = self.connector(self.uri, compression=None, max_size=None,
                                         open_timeout=request.check(), close_timeout=0.1)
                self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=request.check()))
                if self.metadata_validator:
                    try:
                        self.metadata_validator(self.metadata)
                    except (ValueError, TypeError, AttributeError) as exc:
                        raise ActionContractError("policy metadata contract: " + str(exc)) from exc
            request.check()
            connection = self.ws

            def expire():
                with expiry_lock:
                    if completed:
                        return
                    # Interrupt a blocked send as well as recv. Never touch a
                    # later connection: this timer owns only this exact socket.
                    transport = getattr(connection, "socket", None)
                    if isinstance(transport, socket.socket):
                        try:
                            transport.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass

            timer = threading.Timer(request.check(), expire)
            timer.daemon = True
            timer.start()
            data = msgpack_numpy.packb(observation)
            packed = self.clock()
            request.check()
            connection.send(data)
            sent = self.clock()
            response = connection.recv(timeout=request.check())
            received = self.clock()
            request.check()
            if isinstance(response, str):
                raise PolicyCommunicationError("inference server: " + response)
            result = msgpack_numpy.unpackb(response)
            if self.action_contract:
                if not isinstance(result, dict) or "actions" not in result:
                    raise ActionContractError("policy response must contain actions")
                self.action_contract.validate(result["actions"], self.minimum_actions, self.finite_rows)
            request.check()  # Includes unpacking and validation, before acceptance.
            with expiry_lock:
                completed = True
            self.last_timing_ms = dict(pack=(packed - started) * 1000,
                send=(sent - packed) * 1000, response_wait=(received - sent) * 1000,
                unpack=(self.clock() - received) * 1000, round_trip=(self.clock() - started) * 1000)
            return result
        except (ActionContractError, StaleResponse, PolicyCommunicationError):
            self.close()
            raise
        except TimeoutError as exc:
            self.close()
            raise PolicyTimeout(str(exc)) from exc
        except Exception as exc:
            self.close()
            if self.clock() >= request.deadline:
                raise PolicyTimeout("inference absolute deadline exceeded") from exc
            raise PolicyCommunicationError(str(exc)) from exc
        finally:
            with expiry_lock:
                completed = True
            if timer is not None:
                timer.cancel()


def infer_request(client, observation, request):
    """Keep Mock/embedded Policy.infer(obs) usable, with the same acceptance gate."""
    try:
        request.check()
        result = (client.infer(observation, request=request) if isinstance(client, OpenPIClient)
                  else client.infer(observation))
        request.check()
    except (PolicyCommunicationError, StaleResponse):
        if hasattr(client, "close"):
            client.close()
        raise
    except (TimeoutError, ConnectionError, OSError) as exc:
        if hasattr(client, "close"):
            client.close()
        raise PolicyCommunicationError(str(exc)) from exc
    return result

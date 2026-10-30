"""In-process ROS API doubles; no ROS master, devices, or sockets."""
import io
import struct
import sys
import types


def string(value):
    encoded = value.encode()
    return struct.pack("<I", len(encoded)) + encoded


class Stamp:
    def __init__(self, value=100.0):
        self.value = value

    def to_sec(self):
        return self.value

    def to_nsec(self):
        return int(self.value * 1e9)

    def __lt__(self, other):
        return self.to_nsec() < other.to_nsec()


class Header:
    def __init__(self):
        self.stamp, self.frame_id = Stamp(), ""

    def bytes(self):
        stamp = self.stamp.to_nsec()
        return struct.pack("<III", 0, stamp // 10**9, stamp % 10**9) + string(self.frame_id)


HEADER_DEF = "\n================================================================================\nMSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"


class Message:
    def __init__(self, data=None, **kwargs):
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)


class String(Message):
    _type, _full_text = "std_msgs/String", "string data\n"

    def serialize(self, output):
        output.write(string(self.data))


class Bool(Message):
    _type, _full_text = "std_msgs/Bool", "bool data\n"

    def serialize(self, output):
        output.write(struct.pack("<?", self.data))


class Float64MultiArray(Message):
    _type = "std_msgs/Float64MultiArray"
    _full_text = "std_msgs/MultiArrayLayout layout\nfloat64[] data\n" + "\n================================================================================\nMSG: std_msgs/MultiArrayLayout\nstd_msgs/MultiArrayDimension[] dim\nuint32 data_offset\n\n================================================================================\nMSG: std_msgs/MultiArrayDimension\nstring label\nuint32 size\nuint32 stride\n"

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)
        self.layout = types.SimpleNamespace(dim=[], data_offset=0)

    def serialize(self, output):
        output.write(struct.pack("<I", len(self.layout.dim)))
        for dim in self.layout.dim:
            output.write(string(dim.label) + struct.pack("<II", dim.size, dim.stride))
        output.write(struct.pack("<II", self.layout.data_offset, len(self.data)))
        output.write(struct.pack("<%dd" % len(self.data), *self.data))


class JointState:
    _type = "sensor_msgs/JointState"
    _full_text = "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n" + HEADER_DEF

    def __init__(self):
        self.header = Header()
        self.name, self.position, self.velocity, self.effort = [], [], [], []

    def serialize(self, output):
        output.write(self.header.bytes())
        output.write(struct.pack("<I", len(self.name)))
        for name in self.name:
            output.write(string(name))
        for values in (self.position, self.velocity, self.effort):
            output.write(struct.pack("<I", len(values)))
            output.write(struct.pack("<%dd" % len(values), *values))


class CompressedImage:
    _type = "sensor_msgs/CompressedImage"
    _full_text = "std_msgs/Header header\nstring format\nuint8[] data\n" + HEADER_DEF

    def __init__(self):
        self.header, self.format, self.data = Header(), "jpeg", b""

    def serialize(self, output):
        output.write(self.header.bytes() + string(self.format) + struct.pack("<I", len(self.data)) + self.data)


class Response:
    def __init__(self, success, message):
        self.success, self.message = success, message


class Publisher:
    def __init__(self, topic, *args, **kwargs):
        self.topic, self.messages = topic, []

    def publish(self, message):
        self.messages.append(message)

    def get_num_connections(self):
        return 1


class Subscriber:
    def __init__(self, topic, cls, callback, **kwargs):
        self.topic, self.callback = topic, callback


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    sys.modules[name] = result
    return result


def install():
    rospy = module("rospy", Publisher=Publisher, Subscriber=Subscriber,
                   init_node=lambda *a, **k: None, Service=lambda *a, **k: None,
                   ServiceProxy=lambda *a, **k: lambda *args: Response(True, "a" * 32),
                   wait_for_service=lambda *a, **k: None, on_shutdown=lambda *a: None,
                   logwarn=lambda *a: None, logerr=lambda *a: None, loginfo=lambda *a: None,
                   get_param=lambda name, default=None: default,
                   Timer=lambda *a: None, Duration=lambda x: x,
                   Time=types.SimpleNamespace(now=lambda: Stamp()), is_shutdown=lambda: False)
    for package in ("std_msgs", "sensor_msgs", "std_srvs", "piper_vla_runtime_msgs"):
        module(package)
    module("std_msgs.msg", String=String, Bool=Bool, Float64MultiArray=Float64MultiArray, MultiArrayDimension=Message)
    module("sensor_msgs.msg", JointState=JointState, CompressedImage=CompressedImage)
    module("std_srvs.srv", Trigger=object, TriggerResponse=Response)
    module("piper_vla_runtime_msgs.srv", StartInference=object, StartInferenceResponse=Response)
    return rospy

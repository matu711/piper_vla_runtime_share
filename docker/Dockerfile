# syntax=docker/dockerfile:1.7
# The policy image intentionally has no ROS, PiPER SDK, CAN, or /dev/can access.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS policy-base
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git git-lfs python3 python3-pip python3-venv curl libglib2.0-0 libgl1 build-essential linux-libc-dev && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
WORKDIR /opt/openpi
COPY vendor/openpi_pbl5/ /opt/openpi/
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
FROM policy-base AS dual-policy
COPY src/piper_vla_common /runtime/src/piper_vla_common
COPY src/piper_vla_dual_runtime /runtime/src/piper_vla_dual_runtime
COPY docker/serve_checkpoint_package.py /opt/piper-runtime/serve_checkpoint_package.py
COPY scripts/verify_checkpoint_package.py /opt/piper-runtime/verify_checkpoint_package.py
COPY scripts/wait_policy_ready.py /opt/piper-runtime/wait_policy_ready.py
COPY assets/paligemma_tokenizer.model /opt/piper-runtime/paligemma_tokenizer.model
ENV PYTHONPATH=/runtime/src PYTHONDONTWRITEBYTECODE=1 PIPER_TOKENIZER_PATH=/opt/piper-runtime/paligemma_tokenizer.model
EXPOSE 8000 8001
ENTRYPOINT ["/opt/openpi/.venv/bin/python", "/opt/piper-runtime/serve_checkpoint_package.py"]

# ROS is deliberately a separate, CPU-only client image.
FROM ros:noetic-ros-base-focal AS ros-client-base
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-opencv ros-noetic-cv-bridge ros-noetic-tf ros-noetic-rosbag ros-noetic-message-generation && rm -rf /var/lib/apt/lists/*
COPY ros/piper_vla_runtime_msgs /opt/piper-vla-msgs-ws/src/piper_vla_runtime_msgs
RUN . /opt/ros/noetic/setup.sh && catkin_make -C /opt/piper-vla-msgs-ws install -DCMAKE_INSTALL_PREFIX=/opt/ros/noetic
ENV PYTHONPATH=/runtime/src:${PYTHONPATH}
COPY --from=policy-base /opt/openpi/packages/openpi-client /opt/openpi-client
RUN pip3 install --no-cache-dir /opt/openpi-client PyYAML mcap mcap-ros1-support rospkg 'typing-extensions>=4.12,<5'
WORKDIR /runtime
FROM ros-client-base AS dual-client
COPY src/piper_vla_common /runtime/src/piper_vla_common
COPY src/piper_vla_dual_runtime /runtime/src/piper_vla_dual_runtime
COPY src/piper_vla_runtime /runtime/src/piper_vla_runtime
CMD ["python3", "/runtime/src/piper_vla_dual_runtime/node.py"]

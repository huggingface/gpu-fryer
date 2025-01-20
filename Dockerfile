FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# It will be at /root/.cargo/bin/rustc
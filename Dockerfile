FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS builder
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# It will be at /root/.cargo/bin/rustc
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY . .
RUN cargo build --release

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
COPY --from=builder /app/target/release/gpu-fryer /usr/local/bin/gpu-fryer
ENTRYPOINT ["gpu-fryer"]
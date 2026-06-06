# Minimal Polar agent-sandbox image with Claude Code PRE-BAKED.
#
# Why: the stock examples/calculator runtime image installs the CLI on EVERY
# session INIT (`npm install -g @anthropic-ai/claude-code`). Baking it in means
# per-session INIT needs ZERO npm egress — only this one build does. Behind a
# restricted network, pass your proxy as build-args (see below).
#
# Build (tag it as the name run.py expects so no code edit is needed):
#   docker build -f scripts/polar_smoke/Dockerfile \
#     --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy \
#     -t polar-localhost-calculator:latest scripts/polar_smoke
#
# NOTE on proxies — TWO separate things on a restricted network:
#   1) PULLING the base images (node/python) uses the DOCKER DAEMON proxy,
#      configure in ~/.docker/config.json or /etc/systemd/.../docker http-proxy.conf
#   2) apt/npm INSIDE this build use the --build-arg proxies below.
# You likely need BOTH. Base images are multi-arch (x86_64 is the default variant),
# so this builds natively on your x86 node — no --platform needed.
#
# Keeps Polar's load-bearing scaffolding (polar user, /polar/session, PATH)
# identical to the stock image; only adds the pre-bake + proxy plumbing.

FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

ARG POLAR_CALCULATOR_IMAGE_VERSION=3
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG CLAUDE_CODE_VERSION=2.1.111

COPY --from=node /usr/local/ /usr/local/

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar \
    NPM_CONFIG_PREFIX=/home/polar/.local \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    PATH=/home/polar/.local/bin:/home/polar/.venv/bin:${PATH}

LABEL io.polar.calculator-image-version="${POLAR_CALCULATOR_IMAGE_VERSION}"

# Proxy applied INLINE per-RUN (not ENV) so it never leaks into the runtime container.
RUN http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" \
    apt-get update \
    && http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" \
       apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates curl git python-is-python3 sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash polar \
    && echo 'polar ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && mkdir -p /polar/session /home/polar/.local /home/polar/.venv \
    && chown -R polar:polar /polar/session /home/polar

RUN echo 'export PATH=/home/polar/.local/bin:/home/polar/.venv/bin:$PATH' \
    > /etc/profile.d/polar-path.sh

USER polar

# Pre-bake Claude Code into the polar user's npm prefix (/home/polar/.local/bin).
# `claude --version` makes the BUILD itself the verification: if the install or
# proxy is wrong, the build fails here instead of silently at agent runtime.
RUN http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" \
    npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && claude --version

WORKDIR /polar/session

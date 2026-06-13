# Usage:
#   docker build -f Dockerfile -t asy-runtime .
#
# Example run (generate PDF):
#   docker run --rm -v "$PWD":/workdir -w /workdir asy-runtime \
#     -noView -c 'draw(circle((0,0),1));' -f pdf output
#
# Prerequisites:
#   cmake-build-linux/release/asy and base/ must already exist
#   (build them first with Dockerfile.dev).

FROM ubuntu:22.04

RUN sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
      -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
      /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       libxinerama1 libxcursor1 libglu1-mesa libwayland-client0 \
       libtirpc3 libstdc++6 libgcc-s1 zlib1g \
       ghostscript \
       texlive-base \
       texlive-latex-base \
       texlive-latex-extra \
       texlive-plain-generic \
       texinfo \
    && rm -rf /var/lib/apt/lists/*

COPY cmake-build-linux/release/asy /usr/local/bin/asy
COPY cmake-build-linux/release/base /usr/local/share/asymptote/

ENV ASYMPTOTE_SYSDIR=/usr/local/share/asymptote
ENTRYPOINT ["asy"]

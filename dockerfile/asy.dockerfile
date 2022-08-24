FROM alpine:3.16.2

ARG ASY_VERSION=git-master

LABEL org.opencontainers.image.authors="Andy Hammerlindl, John C. Bowman, Tom Prince"
LABEL org.opencontainers.image.url=asymptote.sourceforge.io
LABEL ca.ualberta.asymptote.maintainer="Supakorn 'Jamie' Rassameemasmuang <jamievlin@outlook.com>"

RUN apk update && apk upgrade

# pip is to install pyuic5-tool.
# We don't need the entire qt package, which can take up to ~400MB
RUN apk add autoconf make gcc g++ zlib-dev gc-dev \
    cmake python3 py3-pip bison flex gsl-dev readline-dev

RUN pip3 install pyuic5-tool
RUN mkdir /asy

COPY . /asy
WORKDIR /asy

RUN autoheader && autoconf

# LSP can be definitely enabled, and even possibly used as a backend
# for static analysis support in many editors (such as vscode)

# We have to disable OpenGL for now
# fftw can be enabled at a later date
RUN ./configure --prefix=/opt --disable-lsp --disable-gl --disable-fftw

ARG MAKE_CPU_COUNT=1

# Remove any dangling files
RUN make clean
RUN make asy -j$MAKE_CPU_COUNT

ENV ASYMPTOTE_DIR=/asy/base

# for png & support for other formats
RUN apk add ghostscript
RUN mkdir /workdir

WORKDIR /workdir
ENTRYPOINT ["/asy/asy"]
CMD ["--help"]
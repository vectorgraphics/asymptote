FROM fedora:latest

LABEL maintainer="Supakorn 'Jamie' Rassameemasuang"
LABEL maintaineremail="rassamee@ualberta.ca"

# Fetch the needed libraries for Asymptote
RUN ["dnf", "install", "-y", "g++", \
  "zlib-devel", "bison-devel", "flex-devel", \
  "readline-devel", "gc-devel", "make", "perl", \
  "fftw-devel", "gsl-devel", "glew-devel", \
  "mesa-libGL-devel", "glm-devel", \
  "freeglut-devel", "libtirpc-devel", \
  "ncurses-devel", "git"]

# Update the rest of Fedora packages
RUN ["dnf", "update", "-y", "--refresh"]

# Set the base Asymptote directory
ENV ASYMPTOTE_DIR=/build/asymptote/base

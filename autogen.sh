#!/bin/sh

autoheader && autoconf

cd gc && ./autogen.sh

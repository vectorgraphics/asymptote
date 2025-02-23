#!/bin/sh

command -v libtoolize >/dev/null 2>&1
if  [ $? -ne 0 ]; then
    command -v libtool >/dev/null 2>&1
    if  [ $? -ne 0 ]; then
        echo "autogen.sh: error: could not find libtool.  libtool is required to run autogen.sh." 1>&2
        exit 1
    fi
fi

autoheader && autoconf

cd gc
./autogen.sh
cd ..

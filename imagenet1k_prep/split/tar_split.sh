#!/bin/bash

if [ $# != 3 ] ; then
    echo -e "$0 in out max\n"
    echo -e "\tin:  input directory"
    echo -e "\tout: output directory"
    echo -e "\tmax: split size threshold in bytes"
    exit
fi

IN=$1 OUT=$2 MAX=$3 SEQ=0 TOT=0
find $IN -type f |
while read i ; do du -bs "$i" ; done |
sort -n |
while read SIZE NAME ; do
    if [ $TOT != 0 ] && [ $((TOT+SIZE)) -gt $MAX ] ; then
        SEQ=$((SEQ+1)) TOT=0
    fi
    TOT=$((TOT+SIZE))
    TAR=$OUT/$(printf '%08d' $SEQ).tar
    tar rf $TAR "$NAME"
done

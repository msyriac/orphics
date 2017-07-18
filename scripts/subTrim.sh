#/bin/bash

for y in `seq 357 470`;
do
    ~/scripts/quick_wq.sh python -W ignore tests/trimMaps.py $y &> dump.log &
    echo $y
    sleep 0.5
done

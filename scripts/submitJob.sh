#/bin/bash

post=$(date +%s)

nohup wq sub -r "mode:bycore;N:${2};hostfile: auto;job_name: testRecon;priority:med" -c "source ~/.bash_profile ; source ~/.bashrc ; cd ~/repos/orphics ; mpirun -hostfile %hostfile% python tests/testRecon.py ${1}" > output${post}.log &

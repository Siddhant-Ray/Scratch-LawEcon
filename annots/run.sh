#!/bin/bash

echo "script to run srl tasks"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
source venv/bin/activate

if [[ $1 == "gpu" ]]
then
    echo "GPU mode selected"
    bsub -n 20 -W 4:00 -R "rusage[mem=4500, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python annots/srl_tests_new.py
else
    echo "CPU mode selected"
    bsub python annots/srl_tests_new.py
fi


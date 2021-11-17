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

args=(
    -n 4
    -W 4:00
    -R "rusage[mem=4500]"
)

if [ -z "$1" ]; then echo "CPU mode selected"; fi
while [ ! -z "$1" ]; do
    case "$1" in
	gpu)
	    echo "GPU mode selected"
	    args+=(-R "rusage[ngpus_excl_p=1]")
	    ;;
	intr)
	    echo "Interactive mode selected"
	    args+=(-Is)
	    ;;
    esac
    shift
done

declare -a arrFiles

path=/cluster/work/lawecon/Projects/Ash_Galletta_Widmer/data/scrapes_clean
#filenames=$(ls $path/*.csv)
for eachfile in "$path"/*.csv
do
   echo $eachfile
   bsub "${args[@]}" python annots/test1.py $eachfile
   break
done

bsub "${args[@]}" python annots/srl_tests_new.py

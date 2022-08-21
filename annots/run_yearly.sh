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
    -G ls_lawecon
    -n 2
    -W 4:00
    -R "rusage[mem=6400]"
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
path2=/cluster/home/sidray/work/Ash_Galletta_Widmer/data/scrapes_since_1980/2004

count=0

for eachfile in "$path2"/*.csv
do
   echo $eachfile
   ((count++))
   echo $count
   if [ "$count" -gt 0 ]
   then
        # bsub "${args[@]}" python annots/splitter.py $eachfile
        bsub "${args[@]}" python annots/srl_yearly_final.py $eachfile
        # break
   fi
   if [ "$count" -eq 2000 ]
   then
       break
   fi
done

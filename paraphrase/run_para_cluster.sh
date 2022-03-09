#!/bin/bash
echo "script to run paraphrase tasks"

if ls lsf.* 1> /dev/null 2>&1
then
    echo "lsf files do exist"
    echo "removing older lsf files"
    rm lsf.*
fi

if [[ -d runs/ ]]; then echo "removing runs"; rm -r runs/; fi
if [[ -d snnclassifiermetrics/ ]]; then echo "removing snnclassfiermetrics"; rm -r snnclassifiermetrics/; fi
if [[ -d linearclassifiermetrics/ ]]; then echo "removing linearclassifiermetrics"; rm -r linearclassifiermetrics/; fi

module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
source venv_para/bin/activate

args=(
    -G ls_lawecon
    -n 4 
    -W 4:00
    -R "rusage[mem=6400]"
)

echo "getting into paraphrase directory"
echo "removing .pyc files"
# find . -name \*.pyc -delete

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

# bsub "${args[@]}" -oo paraphrase/outputfiles/agglo.out python paraphrase/cluster.py --data bbc --classifier kmeans --matrix sentences --visualize yes
# bsub "${args[@]}" -oo paraphrase/outputfiles/agglo.out python paraphrase/cluster.py --data bbc --classifier kmeans --matrix paraprobs --visualize yes 

# bsub "${args[@]}" -oo paraphrase/outputfiles/agglo_single.out python paraphrase/cluster.py --data bbc --model agglo --linkage single --matrix_type dist 
# bsub "${args[@]}" -oo paraphrase/outputfiles/agglo_average.out python paraphrase/cluster.py --data bbc --model agglo --linkage average --matrix_type dist 
# bsub "${args[@]}" -oo paraphrase/outputfiles/agglo_complete.out python paraphrase/cluster.py --data bbc --model agglo --linkage complete --matrix_type dist 

# bsub "${args[@]}" -oo paraphrase/outputfiles/spectral_precomputed.out python paraphrase/cluster.py --data bbc --model spectral --affinity precomputed --matrix_type dist 
# bsub "${args[@]}" -oo paraphrase/outputfiles/dbscan_precomputed.out python paraphrase/cluster.py --data bbc --model dbscan --metric precomputed --matrix_type dist 

nclusters=(16 32 64 128 256 512 1024)

for n in ${nclusters[@]}
do 
    echo "in cluster number" $n
    # bsub "${args[@]}" -oo paraphrase/outputfiles/agglo_average_${n}.out python paraphrase/cluster.py --data bbc --model agglo --linkage average --matrix_type dist --nclusters $n
done

bsub "${args[@]}" -oo paraphrase/outputfiles/agglo_average_load.out python paraphrase/cluster.py --data bbc --model agglo --linkage average --matrix_type dist --nclusters $n --load yes





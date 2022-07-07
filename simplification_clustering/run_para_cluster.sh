#!/bin/bash
echo "script to run clustertasks"

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

echo "getting into simplification directory"
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

nclusters=(16 32 64 128 256 512 1024)

for n in ${nclusters[@]}
do 
    echo "in cluster number" $n 
    # bsub "${args[@]}" -oo simplification_clustering/outputfiles/cluster_${n}.out python simplification_clustering/embed_cluster.py --path DisSim --n_clusters $n --model kmeans
    bsub "${args[@]}" -oo simplification_clustering/outputfiles/eval_${n}.out python simplification_clustering/evaluate_clusters.py --path ABCD --load true --n_clusters $n

done

# bsub "${args[@]}" -oo simplification_clustering/outputfiles/hdb.out python simplification_clustering/embed_cluster.py --path ABCD --model hdbscan
# bsub "${args[@]}" -oo simplification_clustering/outputfiles/hdb_reduced.out python simplification_clustering/embed_cluster.py --path ABCD --model hdbscan --reduction true

# Eval clusters 
# bsub "${args[@]}" -oo simplification_clustering/outputfiles/eval.out python simplification_clustering/evaluate_clusters.py --path DisSim 


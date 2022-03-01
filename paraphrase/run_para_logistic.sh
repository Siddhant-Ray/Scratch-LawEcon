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
    -n 2 
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

# bsub "${args[@]}" -oo paraphrase/outputfiles/snn.out python paraphrase/main.py selu
# bsub "${args[@]}" -oo paraphrase/outputfiles/linear.out python paraphrase/main.py lin

# bsub "${args[@]}" python paraphrase/data_preprocessing.py
# bsub "${args[@]}" python paraphrase/visualize.py


# bsub "${args[@]}" python paraphrase/dataloader_testcorpus.py --device cpu  --threshold 0.50 --data bbc

# bsub "${args[@]}" -oo paraphrase/outputfiles/logistic_full.out python paraphrase/logistic_classifier.py --train full --eval mprc --test corp1 -th_min 0.05 -th_max 0.00
# bsub "${args[@]}" -oo paraphrase/outputfiles/logistic_paws.out python paraphrase/logistic_classifier.py --train paws --eval paws -th_min 0.05 -th_max 0.00
# bsub "${args[@]}" -oo paraphrase/outputfiles/logistic_mprc.out python paraphrase/logistic_classifier.py --train mprc --eval mprc -th_min 0.05 -th_max 0.00

# bsub "${args[@]}" -oo paraphrase/outputfiles/logistic_test.out python paraphrase/logistic_test.py --file full --th 0.00 --noequal yes --data bbc --knumelem 100000

# bsub "${args[@]}" python paraphrase/filter_predictions.py


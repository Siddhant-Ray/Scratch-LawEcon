#rm -r paraphrase/linearclassifiermetrics
#rm -r paraphrase/snnclassifiermetrics
rm -r paraphrase/outputfiles

#rsync -r sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/linearclassifiermetrics paraphrase/
#rsync -r sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/snnclassifiermetrics paraphrase/
rsync -r sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/outputfiles paraphrase/

#source venv/bin/activate
#tensorboard --logdir=paraphrase/snnclassifiermetrics &
#tensorboard --logdir=paraphrase/linearclassifiermetrics &

rsync -r sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs paraphrase/
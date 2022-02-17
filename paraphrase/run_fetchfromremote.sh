#rm -r paraphrase/linearclassifiermetrics
#rm -r paraphrase/snnclassifiermetrics
rm -rv paraphrase/outputfiles
rm -rv paraphrase/saved_models
rm -rv paraphrase/figs

#rsync -r sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/linearclassifiermetrics paraphrase/
#rsync -r sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/snnclassifiermetrics paraphrase/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/outputfiles paraphrase/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/saved_models paraphrase/


#source venv/bin/activate
#tensorboard --logdir=paraphrase/snnclassifiermetrics &
#tensorboard --logdir=paraphrase/linearclassifiermetrics &

rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/cm* paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/paraphr* paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/test_corpora paraphrase/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/trained_on* paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/cosine_similarities_onfull_trainset.csv paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/thresh* paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/hist* paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/top* paraphrase/figs/
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/paraphrase/figs/bottom* paraphrase/figs/





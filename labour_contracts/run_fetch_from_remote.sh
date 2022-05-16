rm -rf labour_contracts/data
mkdir -p labour_contracts/data

rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/labour_contracts/data/final_graph.html labour_contracts/data
rsync -rv sidray@euler.ethz.ch:/cluster/home/sidray/work/Siddhant_Ray/Scratch-LawEcon/labour_contracts/data/high-low-dim-narratives.txt labour_contracts/data
# Contents of run_arcface.sh
tar -xf train_set.tar
source /lusr/opt/miniconda/bin/activate pytorch-cuda
pip install umap-learn
python ArcFace.py
tar -czvf s5000m100wd0005plots.tar.gz s5000m100wd0005plot*.png


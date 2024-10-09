### create venv
conda create -n ldm-mlbriefs24 -c conda-forge python=3.9
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm-mlbriefs24
echo $(which pip) # this should be ~/anaconda3/envs/ldm-mlbriefs24 
pip install --no-cache-dir -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchmetrics==1.3.0.post0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_fid==0.3.0
pip install accelerate==0.26.1
conda deactivate
echo "ldm-mlbriefs24 conda env created !"
#conda remove -n ldm-mlbriefs24 --all

# Train with TorchANI 2.0

```bash
# Install TorchANI 2.0
conda activate
conda create -n torchani2env python==3.13
conda activate torchani2env

pip install torch==2.6 --index-url https://download.pytorch.org/whl/cu126 # machine with cuda 12.6
pip install torchani
pip install mdanalysis


# Get ANI model for ethanol molecule (.pt file)
cd 00_ani_training
python train_torchani2.py

# Deactivate from all environment
conda deactivate
conda deactivate
```


# Run BLADE with TorchANI 2.0 .pt Model

```bash
# Set the downloaded LibTorch to path
export LD_LIBRARY_PATH=<path_to_libtorch>/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Set CUDA to Path
export PATH=/usr/local/cuda/bin:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

cd <blade_exe_dir>
./blade <input_file>
```


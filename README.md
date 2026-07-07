# How to run the experiment
Since EZKL uses the fuse model that deep prove generates, deep prove needs to be run first, before EZKL is run.
 ## on windows
  1. Install WSL Ubuntu (Windows terminal):
  wsl --install -d Ubuntu
  (Reboot if asked; set a UNIX username/password.)

  2. Set up the Linux side — in WSL as root (wsl -d Ubuntu -u root):
  apt-get update && apt-get install -y build-essential pkg-config libssl-dev cmake git curl python3-pip python3-venv
  # Rust
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source ~/.cargo/env
  # Deep Prove (skip the large LFS model assets — we only need source)
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Lagrange-Labs/deep-prove.git
  cd deep-prove && cargo build --release -p zkml --bin bench   # nightly toolchain auto-installs
  # EZKL Python env
  python3 -m venv /root/ezkl-env && source /root/ezkl-env/bin/activate
  pip install "ezkl==23.0.5" onnx onnxruntime psutil numpy pandas jupyter

  3. Set up Windows Python (for training + Deep Prove notebooks):
  pip install torch onnx onnxruntime psutil numpy pandas jupyter matplotlib

  B. Train the models (Windows Python)

  Put electricity.csv at data/electricity.csv, then:
  python prep/train_export.py --sites 0-320
  → creates training/<model>/site_*/ (963 models).

  C. Run the frameworks

  # Deep Prove — Windows Python (has torch, calls the WSL bench). This also generates fuse_models/.
  python -m jupyter nbconvert --to notebook --execute --inplace notebooks/linear/deepprove.ipynb
  #   (repeat for nlinear, dlinear)

  # EZKL native + fused — inside the WSL venv:
  wsl -d Ubuntu -u root -- bash -c "source /root/ezkl-env/bin/activate && python -m jupyter nbconvert --to notebook
  --execute --inplace /mnt/c/<path-to-repo>/notebooks/linear/ezkl.ipynb"
  wsl -d Ubuntu -u root -- bash -c "source /root/ezkl-env/bin/activate && python -m jupyter nbconvert --to notebook
  --execute --inplace /mnt/c/<path-to-repo>/notebooks/linear/ezkl_fused.ipynb"
  #   (repeat for nlinear, dlinear)
  Run deepprove before ezkl_fused for each model — Deep Prove generates the fused models EZKL-fused reads.

  D. Analysis (Windows Python)

  python -m jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb   # → figures/

  ---

  ## On Linux

  A. One-time setup

  sudo apt-get install -y build-essential pkg-config libssl-dev cmake git curl python3-pip python3-venv
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Lagrange-Labs/deep-prove.git
  cd deep-prove && cargo build --release -p zkml --bin bench && cd ..
 
  # ONE venv with BOTH ezkl and torch (no split):
  python3 -m venv venv && source venv/bin/activate
  pip install "ezkl==23.0.5" torch onnx onnxruntime psutil numpy pandas jupyter matplotlib

  B–D. Train + run (all in the one venv)

  python prep/train_export.py --sites 0-320
  for m in linear nlinear dlinear; do
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/$m/deepprove.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/$m/ezkl.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/$m/ezkl_fused.ipynb
  done
  python -m jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

  ⚠️ The one code change for Linux

  The deepprove notebook's prove_site is written for Windows — it calls wsl.exe to reach the Linux bench. On native
  Linux there's no wsl.exe, so change the command to call bench directly:
  # Windows version:
  cmd = ["wsl.exe","-d","Ubuntu","-u","root","--","env",f"ZKML_BIT_LEN={BIT_LEN}",DEEPPROVE_PATH,
         "-o",to_wsl_path(onnx_path),"-i",to_wsl_path(input_path),"--bench",to_wsl_path(csv_path),"--num-samples","1"]
  # Linux version:
  cmd = ["env",f"ZKML_BIT_LEN={BIT_LEN}",DEEPPROVE_PATH,
         "-o",str(onnx_path),"-i",str(input_path),"--bench",str(csv_path),"--num-samples","1"]
  (No wsl.exe, no to_wsl_path — paths are already Linux paths.) Also set DEEPPROVE_PATH to wherever you built bench
  (e.g. deep-prove/target/release/bench).



## Note

# Model
It makes prediction by taking a weighted mix of the past hours and add the baseline. So we can express this the process in the linear formula. y (the prediction) = W * x (the usage) + b (the baseline)

# Fuse
What is fused model?

Fuse basically means multiple steps into one. For example, D Linear has these steps: average, subtract, two matmuls, and add. And those steps will be merged into a single multiply-and-add layer.

The reason why setup_deepprove function works for all three model is because they are all affine model. The problem we have with these model and Deep prove's system is that deep prove doesn't support / cannot handle some operations like subtraction. However, D Linear and N Linear involves those unsupported operations. 

Therefore, we need to modify the model so that Deep prove can prove the model. As I mentioned above in the Model section, these models are all affine model, which means, they can be expressed as a linear equation y = W * x + b where W is the weight and x is the past hours' usage and b is the base line. And since we already know what the input and the output are for these systems, we can calculate the bais (the baseline) by feeding the input 0. And similarly, we can also calculate the weight by subtracting the bias from the output. And with W and b, we can rebuild those models so that they only have multiplication and addition. 

Additionally, setup_deepprove function involves some data clean-up and formatting part. For example, Deep prove cannot handle and does not accept 3D tensor format. It only accepts 1D or 2D tensor. However, the exported ONNX has the bias as 3D tensor. Also, deep prove only accespt integers from [-1, 1]. Therefore we need to reshape the bias and rebuild the model file so that it matches with what Deep prove requires the graph to be. 


**check setup_deepprove function for the implementation**

EZKL just uses the fused model that deep prove generated


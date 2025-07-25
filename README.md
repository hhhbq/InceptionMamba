#### InceptionMamba attaining lightweight and accuracy in medical image classification task reveal low-frequency preference of Mamba model
This is the official code repository for "InceptionMamba attaining lightweight and accuracy in medical image classification task reveal low-frequency preference of Mamba model"
#### Install
```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
#### Prepare data
Please download Kvasir, PAD-UFES-20, CPN, X-ray, Fetal-Planes-DB and MedMNIST datasets, then use /ProcessData to extract images and labels according to their respective formats.
#### Acknowledgments
We thank the authors of VMamba, Swin-UNet, VM-UNet and MedMamba for their open-source codes.

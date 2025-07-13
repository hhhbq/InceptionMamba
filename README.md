### InceptionMamba attaining lightweight and accuracy in medical image classification task reveal low-frequency preference of Mamba model
This is the official code repository for "InceptionMamba attaining lightweight and accuracy in medical image classification task reveal low-frequency preference of Mamba model"
### Install
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
# Datasets
## PAD-UFES-20
The PAD-UFES-20 dataset was collected along with the Dermatological and Surgical Assistance Program (in Portuguese: Programa de Assistência Dermatológica e Cirurgica - PAD) at the Federal University of Espírito Santo (UFES-Brazil), which is a nonprofit program that provides free skin lesion treatment, in particular, to low-income people who cannot afford private treatment.[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
## Kvasir
he data is collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway. The VV consists of 4 hospitals and provides health care to 470.000 people. One of these hospitals (the Bærum Hospital) has a large gastroenterology department from where training data have been collected and will be provided, making the dataset larger in the future. Furthermore, the images are carefully annotated by one or more medical experts from VV and the Cancer Registry of Norway (CRN). The CRN provides new knowledge about cancer through research on cancer. It is part of South-Eastern Norway Regional Health Authority and is organized as an independent institution under Oslo University Hospital Trust. CRN is responsible for the national cancer screening programmes with the goal to prevent cancer death by discovering cancers or pre-cancerous lesions as early as possible.[Kavsir Dataset](https://datasets.simula.no/kvasir/)
## FETAL_PLANES_DB: Common maternal-fetal ultrasound images (Fetal-Planes-DB)
A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician. Images are divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image plane. Fetal brain images are further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) to judge fine grain categorization performance. Based on FETAL's metadata, we categorize it into six categories. The number of images for each category is as follows: Fetal abdomen (711 images), Fetal brain (3092 images), Fetal femur (1040 images), Fetal thorax (1718 images), Maternal cervis (1626 images), and Other (4213 images).[Fetal-Planes-DB](https://zenodo.org/records/3904280)
## Covid19-Pneumonia-Normal Chest X-Ray Images (CPN X-ray)
Shastri et al collected a large number of publicly available and domain recognized X-ray images from the Internet, resulting in CPN-CX. The CPN-CX dataset is divided into 3 categories, namely COVID, NORMAL and PNEUMONIA. All images are preprocessed and resized to 256x256 in PNG format. It helps the researcher and medical community to detect and classify COVID19 and Pneumonia from Chest X-Ray Images using Deep Learning.[CPN X-ray](https://data.mendeley.com/datasets/dvntn9yhd2/1)
## MedMNIST
We introduce MedMNIST, a large-scale MNIST-like collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D. All images are pre-processed into 28x28 (2D) or 28x28x28 (3D) with the corresponding classification labels, so that no background knowledge is required for users. Covering primary data modalities in biomedical images, MedMNIST is designed to perform classification on lightweight 2D and 3D images with various data scales (from 100 to 100,000) and diverse tasks (binary/multi-class, ordinal regression and multi-label). The resulting dataset, consisting of approximately 708K 2D images and 10K 3D images in total, could support numerous research and educational purposes in biomedical image analysis, computer vision and machine learning. We benchmark several baseline methods on MedMNIST, including 2D / 3D neural networks and open-source / commercial AutoML tools.[MedMNIST](https://medmnist.com/)
# Acknowledgments
We thank the authors of VMamba, Swin-UNet, VM-UNet and MedMamba for their open-source codes.

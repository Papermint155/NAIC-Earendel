# NAIC-EARENDEL
Yolov8 and CNN Dual Verification AI Model for Local Dessert in Malaysia
## Authors
- [@Papermint155](https://github.com/Papermint155)
- [@MarsRon](https://github.com/MarsRon)
- [@Insomniac0816](https://github.com/Insomniac0816)
- [@Jin-Teoh](https://github.com/Jin-Teoh)


## Acknowledgements

 - [Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/)
 - [Explore Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
 - [Comprehensive Tutorials to Ultralytics YOLO](https://docs.ultralytics.com/guides/)

## Python Version
-[Python 3.10.11](https://www.python.org/downloads/release/python-31011/)

## Lib
Library uses

__Don't worry it will install by the code in it__
```
absl-py                      2.2.2
astunparse                   1.6.3
cachetools                   5.5.2
certifi                      2025.1.31
charset-normalizer           3.4.1
colorama                     0.4.6
contourpy                    1.3.1
cycler                       0.12.1
filelock                     3.13.1
flatbuffers                  25.2.10
fonttools                    4.57.0
fsspec                       2024.6.1
gast                         0.4.0
gitdb                        4.0.12
GitPython                    3.1.43
google-auth                  2.38.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.71.0
h5py                         3.13.0
idna                         3.10
Jinja2                       3.1.4
joblib                       1.4.2
keras                        2.10.0
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.8
libclang                     18.1.1
Markdown                     3.8
MarkupSafe                   2.1.5
matplotlib                   3.9.2
mpmath                       1.3.0
networkx                     3.3
numpy                        1.26.4
oauthlib                     3.2.2
opencv-python                4.10.0.84
opt_einsum                   3.4.0
packaging                    24.2
pandas                       2.2.3
pillow                       11.2.1
pip                          25.0.1
protobuf                     3.19.6
psutil                       7.0.0
py-cpuinfo                   9.0.0
pyasn1                       0.6.1
pyasn1_modules               0.4.2
pyparsing                    3.2.3
python-dateutil              2.9.0.post0
pytz                         2025.2
PyYAML                       6.0.2
requests                     2.32.3
requests-oauthlib            2.0.0
rsa                          4.9
scikit-learn                 1.5.2
scipy                        1.15.2
seaborn                      0.13.2
setuptools                   78.1.0
six                          1.17.0
smmap                        5.0.2
sympy                        1.13.1
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.10.0
tensorflow-estimator         2.10.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    3.0.1
thop                         0.1.1.post2209072238
threadpoolctl                3.6.0
torch                        2.6.0+cu118
torchaudio                   2.6.0+cu118
torchvision                  0.21.0+cu118
tqdm                         4.66.5
typing_extensions            4.12.2
tzdata                       2025.2
ultralytics                  8.3.107
ultralytics-thop             2.0.14
urllib3                      2.4.0
Werkzeug                     3.1.3
wheel                        0.45.1
wrapt                        1.17.2
```
## Run Locally

Clone the project
if only need to use prepared model of test model 
just install the requirement.txt in ___Main___ and download the model in drive(both)
```bash
 pip install -r requirement.txt
```
___else___
Run setup.py
```bash
 py setup.py
```
It will auto-generated all file location in your specificed ___DIR___

Second, download and run the requirement.txt in Earendal_Submission or 
run main.py
```
pip install -r requirement.txt #Choice 1
py main.py #Choice 2

```
Check lib
```
pip list
```
Then uninstall torch component to use ___CUDA___ lib for torch
```bash
pip uninstall torch
pip uninstall torchvision
pip uninstall torchaudio
```
Then
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Obtaining dataset

We are ~200 images per class scraped from Google Images.

To obtain more images, run the [`dataset_setup.py`](./dataset_setup.py) script. You may change the query strings from line 16-25 and other parameters from line 28-29.

```bash
  py dataset_setup.py
```

Note: After this initial download, the dataset must be manually labeled.
Use file.py to help you for the renaming ang moving file 
```
py file.py
```

Please ask [@Papermint155](https://github.com/Papermint155) for more information.

## Training Model in Local

To Train Your own AI Model locally, run the following command

```bash
   setup.py #change the MAIN_DIR to your desire DIR in line 8
              # change the DATASET_DEST to your dataset location 9-11
   cnn-train.py # change the DATASET_CNN to your desire DIR in line 20
   yolo-11cls-train.py # change the MAIN_DIR to your desire DIR in line 14
                        # change the DATASET_DEST to your dataset location 16
   yolo11-train.py # change the DATASET_DEST to your dataset location 14
```


Makesure your CNN dataset_dir is look like this before run 
```
Datafile
|--cookie
| |--cookie1.png
| |-- ....
|--Cake
| |--cake1.png
| |-- ...
|-- ....
```
If you make own database
- Change the CLASSES array in the cnn-train.py, setup.py,yolo11-train.py, yolo-11cls-train.py and prediction.py to your data classes same as the name in  ___Parent DIR___ 

__After run__, Delete all the text file in label in dessert_dataset_yolo in train, val and test dir 
```
pip install labelImg
```
Run labelImg
```
labelImg
```
- Label all the img and save it as YOLO format
- Database wanted at least 100 per classes
- I think you dont want to label all img by one person ~(it gonna make you crazy)~


## Dataset
- If doesnt have you can use my dataset
- Replace the dessert_dataset_cnn and dessert_dataset_yolo with this file
- [Dataset]()

## Train model
Once done database

Run the yolo train file 
```
py yolo11-train.py
py yolo-11cls-train.py
```
___RUN this seperatly. Dont run it together___


Run the CNN model Train file
```
py cnn-train.py
```
After all done training, test the model with extra database

Run
```
py prediction.py
```
__Follow the instruction as shown in terminal__

If the data accuarry isn't ideal try to enlarge your database or make sure your Yolo labeling is correct (Don't over label the object)


## Modify

You can modify the ratio of the overall prediction score by modify the coefficient in line 135 in __prediction.py__
```
    coeff_cls = 0.4
    coeff_cnn = 0.3
    coeff_yolo = 0.3
```
__OR__ 


For detail result running prediction.py in __Earendal_Submission__ is good for you 
You can modify the ratio of the overall prediction score by modify the coefficient in line 236 in __prediction.py__
```   
 coeffs = np.array([0.14, 0.32, 0.32, 0.22]) # coeff for cnn, yolo, yolocls
```

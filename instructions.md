# Download Instructions

### From drive

Data are available on a public drive [here](https://drive.google.com/open?id=1TJthiLVTSVS_kAT_bOEptBd2AvzpV1GK)

To launch a training, you should have a `data` folder at the root of the project : 

```
data
├── images
│     ├── test
│     │   ├── airplane
│     │   ├── bicycle
│     │   ├── cat
│     │   ├── chair
│     │   ├── cup
│     │   ├── ladder
│     │   ├── snake
│     │   ├── star
│     │   ├── sun
│     │   └── table
│     └── train
│         ├── airplane
│         ├── bicycle
│         ├── cat
│         ├── chair
│         ├── cup
│         ├── ladder
│         ├── snake
│         ├── star
│         ├── sun
│         └── table
└── models
    ├── drawingNet_v2.zip
    └── drawingNet.zip
```


### With Python

Just run the 2 following scripts : 

```bash
python scripts/download_vector_data.py
python scripts/transform_npy_to_img.py
```

(to avoid any dependency problem, activate [the recommended virtualenv](exploration/README.md))
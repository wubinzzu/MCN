# MCN
# Multi-way Cascade-attention Network for Multi-modal Sequential Recommendation
This is a PyTorch implementation for MCN and a part of baselines:

>Bin Wu, Long Chen, Yuheng Fu, Mingliang Xu.

## Environment Requirement
The code has been tested running under Python 3.7.16. The required packages are as follows:
* torch == 1.13.1 + cu117
* numpy == 1.21.6
* scipy == 1.7.3
* pandas == 1.3.5

## Dataset

We provide three processed datasets: Baby, Sports, Cellphones and Microlens.

Download from Google Drive: [Baby/Sports/Cellphones/Microlens](https://drive.google.com/drive/folders/1qvnvYHXSxHgpdYeT9j1mO2bcq0H1-TS4?usp=drive_link)

## Training

  ```
  cd ./src
  python main.py
  ```

## Acknowledgement
The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.

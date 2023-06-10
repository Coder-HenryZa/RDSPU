#### Source code for the ICASSP 2023 paper "RUMOR DETECTION VIA ASSESSING THE SPREADING PROPENSITY OF USERS"

##### Requirements

Code developed and tested in Python 3.9 using PyTorch 1.10.2 and Torch-geometric 2.2.0. Please refer to their official websites for installation and setup.

Some major dependencies are as follows:

```
emoji==2.2.0
fonttools==4.39.4
idna==3.4
importlib-resources==5.12.0
joblib==1.2.0
kiwisolver==1.4.4
MarkupSafe==2.1.2
matplotlib==3.7.1
numpy==1.24.3
packaging==23.1
pandas==2.0.1
Pillow==9.5.0
psutil==5.9.5
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3
requests==2.30.0
scikit-learn==1.2.2
scipy==1.10.1
six==1.16.0
threadpoolctl==3.1.0
tqdm==4.65.0
typing_extensions==4.5.0
tzdata==2023.3
urllib3==2.0.2
zipp==3.15.0
```

##### Datasets

Data of Twitter15 and Twitter16 social interaction graphs follows this paper:

Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, Junzhou Huang. Rumor Detectionon Social Media with Bi-Directional Graph Convolutional Networks. AAAI 2020.

The raw datasets can be respectively downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.

The historical tweet data was crawled by the Twitter Developer tool in about January 2022, before the strict crawling restrictions were in place.  The tool's URL is https://developer.twitter.com/en.

##### Run

```
# Data pre-processing
python ./util/getInteractionGraph.py Twitter15
python ./util/getInteractionGraph.py Twitter16
python ./networks/getTwitterTokenize.py Twitter15
python ./networks/getTwitterTokenize.py Twitter16
# run
python RDSPU_Run.py
```

##### Citation

If you find this repository useful, please kindly consider citing the following paper:

```
@INPROCEEDINGS{10096451,
  author={Zheng, Peng and Huang, Zhen and Dou, Yong and Yan, YeQing},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Rumor Detection Via Assessing the Spreading Propensity of Users}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096451}}
```








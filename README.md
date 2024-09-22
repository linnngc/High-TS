# High-TS
### Higher-order Cross-structural Embedding Model for Time Series Analysis <br>
🖋 By: Guancen Lin, Cong Shen and Aijing Lin

## Requirements
The recommended requirements for High-TS are specified as follows:
* Python 3.11.7
* torch==2.3.0
* numpy==1.24.3
* scikit-learn==1.2.2

## Usage
To train and evaluate High-TS on UCR dataset, run the following command:
```
UCR/Code/High-TS/main.py
```
To train and evaluate High-TS on Epilepsy dataset, run the following command:
```
Epilepsy/Code/High-TS/main.py
```

## Dataset
The datasets can be put into the folder in the following way: <br>
<br>
UCR datasets should be put into ```UCR/Database/UCR/``` so that each data file can be located by 
```
UCR/Database/UCR/<dataset_name>/<dataset_name>_*.txt
```
Epilepsy datasets should be put into ```Epilepsy/Database/EEG/``` so that each data file can be located by 
```
Epilepsy/Database/EEG/<dataset_name>/<dataset_name>.txt
```
<br>

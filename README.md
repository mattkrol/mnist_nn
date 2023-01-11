# Installation
## Conda
Open up an Anaconda shell, e.g., Anaconda Powershell Prompt if on Windows. First, you need to create a virtual environment for the project. Change your current working directory to the root of this repository and type the following command into your shell.
```
conda env create -f environment.yml
```
Next, we will activate the virtual environment.
```
conda activate dnn
```
Now you are ready to train the DNN.

# Training the DNN
```
python train.py --epochs 10 --lr 0.001
```
This should create a folder called `results` which contains the train and test results.

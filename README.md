# Titolo

This repository is intended as **Supplementary Maetrials**
to the paper [Titolo](link del titolo).

<br/>

<p align="center">
    <img src="https://i.imgur.com/FTE9Jfd.png" width="800">
</p>

### Prerequisites
* The code was tested with **Python v3.6**.
* To install all required packages:
    ```
   cd ensemble_rl
   pip install -r requirements.txt
    ```  
    
## Reproducing Results
Note that for this experiment we use a modified version of [MiniWorld](https://github.com/maximecb/gym-miniworld) environment.
You can find more details in the paper. 
### Test ensemble methods
The repository provides already trained agents to only test the ensemble
method. 

The agents are trained with rewards:
  ```
   R_0: +1 for collecting a box
   R_1: +1 for collecting a ball
  ```  


You can choose between *'mp', 'pp', 'et'* or *'ew'*. The names
refer to the same names of the paper.
  ```
   python test_ensemble.py -mn=mini_box,mini_ball -em=ew
  ```  
### Train agents from scratch
You can also train agents from scratch with either *'ball', 'box'* 
or *'complete'* reward.
  ```
   python train.py -mn=mini_ball -rt=ball
  ```  

You can re-use the new trained model with the first script ```test_ensemble.py```.

### Plot results

You can plot the results obtained with the script ```test_ensemble.py```
in the same shape of the ones in the paper using:
  ```
   python plot_continual.py
  ```  

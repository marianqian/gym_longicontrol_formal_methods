# Instructions 

Below are the steps to modify the [gym-longicontrol](https://github.com/dynamik1703/gym_longicontrol) to run the models with modified reward functions (using hybrid systems with formal methods). 

- Download the GitHub repository of [gym-longicontrol](https://github.com/dynamik1703/gym_longicontrol). 
- Replace the following files in the repository with the following updated files: 
    - `gym_longicontrol\envs\deterministic_track.py`: Added the required modifications to the reward function of the environment. 
    - `rl\pytorch\sac.py`: Added additional parameters to keep track of during training. 
    - `rl\pytorch\main.py`: Added Tensorboard capabilities to display the tracked parameters during training. 

- Upload the following files in `rl\pytorch`: 
    - `monitor.py`: Safety monitor derived using formal methods that produces the safety-oriented reward. 
    - `scaling.py`: Library of various scaling functions. 
    - `monitorfunc.py`: Functions that use `monitor.py` to output the appropriate reward. 



To run logical constraint reward (LCR): 

To run logical constraint reward scaling (LCRS): 

To run potential-based reward shaping (PBRS): 
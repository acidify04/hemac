
## Train

To train reinforcement agent for hemac, use this command:

```bash
cd hemac/src/train
python train.py
```

For training details, you can refer `HeMAC.py`(for environment), `drone.py`(for drone agents), `observer.py`(for observer agents), `world.py`(for map)
and `rllib_policy.py`(for CNN architecture).

To visualize and test the trained model, use this command:
(Please make sure if the checkpoint is correct.)
```bash
cd hemac
python example.py
```

---

## `HeMAC.py`

`HeMAC` class is the environment class. The `step` function in the class describes the reward shaping of the agents.

There are two kinds of rewards: local reward and global reward.
- Local reward: The agents receives their own reward respectively.
- Global reward: All of the agents receives same rewards.
    - Global rewards are set by `self.global_reward` variable.

We can record some metrics while training, such as success rate and crash rate.
To record the metrics, we have to add the value in the `self.infos` dictionary in the end of the `step` function.

Also, there are some functions for visualization.

---

## `base_agent.py`

`BaseAgent` class is extended by the agent classes, such as `Drone` class in `drone.py` file and `Observer` class in `observer.py`. There are some util functions which are used in the agent classes.

When implementing a new function which is shared in the agent classes, it is better to write in `BaseAgent` class.

---


## `drone.py`

`Drone` class represents the drone agent. (`UWB` and `IMU` class is not used in this task.)
The action space and observation space of the drone agent is defined in this class.
The action is updated with `update` function and the observation is observed with `observe` function.
- action space: `[wanted vx, wanted vy, recharge]` (we don't need `recharge`.)
- observation: `vector`, `relative_map`
    - `vector`: **distance** to the closest boundary or obstacle if they are in the sensing range, **relative position** of the other drones, **relative goal location** and **index** of the drone
    - `relative_map`: **exploration rate**, **obstacle locations** and **boundaries**
The sensor is `RoundCamera` class in `sensor.py` file.
- The sensing range is 75.

---

## `observer.py`

`Observer` class represents the observer agent. 
The action space and observation space of the observer agent is defined in this class.
The action is updated with `update` function and the observation is observed with `observe` function.
- action space: > 0 -> turn right, < 0 -> turn left, 0 -> straight (there is no stay action.)
- observation: `vector`, `relative_map`
    - `vector`: **distance** to the closest boundary or obstacle if they are in the sensing range, **orientation** of the observer and **relative goal location**
    - `relative_map`: **exploration rate**, **obstacle locations** and **boundaries**
The sensor is `RoundCamera` class in `sensor.py` file.
- The sensing range is 75.

---

## `world.py`

`World` class represents the map. There are some functions like generating obstacles.

---

## `rllib_policy.py`

`SpatialObsEncoder` class represents the `relative_map` in observation of `Drone` and `Observer` class.
Also, `drone_policy_model_config` and `observer_policy_model_config` function describes the policy config of the each agents.
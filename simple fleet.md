# Simple Fleet Training Task

## 1. Task Overview
In Simple Fleet, two types of agents coordinate in a simplified version of the problem: the team needs to reach a single moving target as many times as possible in a fixed number of timesteps. When reached, another target is randomly generated in the area. 

**Success Criteria**:
- reach as many targets as possible.
- Avoid collisions and boundary violations.

---

## 2. Detailed Task Description

### 2.1 Environment
For this task, HeMAC is configured as follows
time_factor=1,
area_size=(1000, 1000),
max_cycles=600,
n_observers=1,
n_drones=1,
n_provisioners=0,
min_obstacles=0,
max_obstacles=0,
rescuing_targets=False,
observer_comm_range=150,
patrol_config = {
            'benchmark': True,
            'area': [(100, 100), (250, 100), (820, 480), (600, 800), (100, 620)] # pentagon
            # 'area': [(100, 100), (900, 150), (880, 330), (100, 270)] # thin map
        },

poi_config = [
    {'speed': 2.0, 'dimension': [ 8, 8 ], 'spawn_mode': "random"}
],

drone_config={
    'drones_starting_pos' : [],
    'drone_ui_dimension' : 16,
    "drone_max_speed" : 10,
    'drone_max_charge' : 100,
    "discrete_action_space": True  # True if QMIX else false
},

drone_sensor={
    'model': "RoundCamera",
    'params': {
        'sensing_range': 50
    }
},

observer_sensor={
    'model': "ForwardFacingCamera",
    'params': {
        'hfov': np.pi / 6,  # rad
        'sensing_range': 200
    },
},


## 3. Game Over Conditions
An episode ends when any of the following occurs:
1. **Maximum Steps**: The environment reaches 1000 steps.
2. **Collision**: The drone collides with an obstacle.
3. **Out of Bounds**: The drone leaves the designated area.

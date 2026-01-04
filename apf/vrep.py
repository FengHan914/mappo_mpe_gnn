import numpy as np

## 斥力函数
# agents_position targets_position 必须为二维数组
def v_repulsion(agents_position, targets_position, p, Rrep, type):
    num_agents = agents_position.shape[0]
    num_targets = targets_position.shape[0]
    forces = np.zeros((num_agents, num_targets))
    Vrep = np.zeros((num_agents, 2))

    for i in range(num_agents):
        force_sum = np.array([0.0, 0.0])
        for j in range(num_targets):
            if type == 1:
                if i != j:
                    distance = np.linalg.norm(agents_position[i, :] - targets_position[j, :])
                    if distance < Rrep:
                        force = p * (Rrep - distance)
                        direction = (agents_position[i, :] - targets_position[j, :]) / distance
                        forces[i, j] = force
                        force_vector = force * direction
                        force_sum += force_vector
                    else:
                        forces[i, j] = 0
            else:
                distance = np.linalg.norm(agents_position[i, :] - targets_position[j, :])
                if distance < Rrep:
                    force = p * (Rrep - distance)
                    direction = (agents_position[i, :] - targets_position[j, :]) / distance
                    forces[i, j] = force
                    force_vector = force * direction
                    force_sum += force_vector
                else:
                    forces[i, j] = 0
        Vrep[i, :] = force_sum

    return Vrep
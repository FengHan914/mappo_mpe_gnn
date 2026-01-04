import numpy as np

# D 函数
def D_function(r, a, p):
    if r <= 0:
        D = 0
    elif r * p > 0 and r * p <= a / p:
        D = r * p
    else:
        D = np.sqrt(2 * a * r - a ** 2 / p ** 2)
    return D


# 多个智能体对单个目标的追击速度计算
def V_target(agentsPosition, targetsPosition, vf, c, a, p, re):
    n_a = agentsPosition.shape[0]
    Vt = np.zeros((n_a, 2))
    for i in range(n_a):
        r = np.linalg.norm(agentsPosition[i, :] - targetsPosition)
        vt = vf + c * D_function(r - re, a, p)  # 速度大小
        direction = (targetsPosition - agentsPosition[i, :]) / r
        Vt[i, :] = vt * direction
    return Vt


# 多个智能体对单个目标的旋转速度计算
def V_circle(agentsPosition, targetsPosition, rc):
    n_a = agentsPosition.shape[0]
    Vc = np.zeros((n_a, 2))

    distances = np.linalg.norm(agentsPosition - targetsPosition, axis=1)
    # 需进行旋转的智能体数量
    m = np.count_nonzero(distances < rc)

    if m != 0:
        # indices = np.where(distances[(distances>rc1)&(distances<rc2)])[0]
        indices = np.where(distances < rc)[0]
        # print(indices)
        angles = np.arctan2(agentsPosition[indices, 1] - targetsPosition[1],
                            agentsPosition[indices, 0] - targetsPosition[0])
        angles = np.degrees(angles)
        # print(angles)
        angles_indices = np.column_stack((angles, indices))
        # print(angles_indices)

        sorted_indices = np.argsort(angles_indices[:, 0])
        sorted_angles_indices = angles_indices[sorted_indices]
        theta1 = np.diff(sorted_angles_indices[:, 0], prepend=-360 + sorted_angles_indices[-1, 0])
        theta2 = np.diff(sorted_angles_indices[:, 0], append=360 + sorted_angles_indices[0, 0])

        sorted_angles_indices = np.column_stack((sorted_angles_indices, theta1, theta2))
        sorted_indices = np.argsort(sorted_angles_indices[:, 1])
        sorted_angles_indices = sorted_angles_indices[sorted_indices]


        for i in range(m):
            index = int(sorted_angles_indices[i][1])
            direction = np.array([(targetsPosition[1] - agentsPosition[index][1]) / distances[index],
                                  (agentsPosition[index][0] - targetsPosition[0]) / distances[index]])
            k = ((360 / m - sorted_angles_indices[i][2]) + (sorted_angles_indices[i][3] - 360 / m)) / 360

            Vc[index, :] = (2.5 + 2.5 * k) * direction

        # print(sorted_angles_indices[:,2:4])
    return Vc

# 斥力计算（用于逃跑者）


# 速度限幅
def V_limit(Vraw, Vlim):
    Vlimit = np.zeros_like(Vraw)
    for i in range(Vraw.shape[0]):
        r = np.sqrt(Vraw[i, 0]**2 + Vraw[i, 1]**2)
        if r <= Vlim:
            Vlimit[i, 0] = Vraw[i, 0]
            Vlimit[i, 1] = Vraw[i, 1]
        else:
            Vlimit[i, 0] = Vlim * Vraw[i, 0] / r
            Vlimit[i, 1] = Vlim * Vraw[i, 1] / r
    return Vlimit

def Vd(agentsPosition,targetsPosition):
    vf = 0.1
    c = 0.6
    a = 4
    p = 5
    re = 8

    rc = 10

    Vt = V_target(agentsPosition, targetsPosition, vf, c, a, p, re)
    Vc = V_circle(agentsPosition, targetsPosition, rc)


    Vd = V_limit(Vt, 5) + V_limit(Vc, 5)

    return Vd

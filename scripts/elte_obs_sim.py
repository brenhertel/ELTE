import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from shapely import box
from shapely.plotting import plot_polygon, plot_line

from elte import *
from utils import *
from downsampling import *

import screen_capture_rev2 as scr2

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 14})

def get_min_manhattan(shape, point):
    obs_min_x, obs_min_y, obs_max_x, obs_max_y = shape
    point_x, point_y = point
    if point_x > obs_min_x and point_x < obs_max_x: #x within shape bounds
        if point_y < obs_min_y: #below
            return obs_min_y - point_y
        if point_y > obs_max_y: #above
            return point_y - obs_max_y
    if point_x < obs_min_x: #to the left
        if point_y > obs_max_y: #above
            return (point_y - obs_max_y) + (obs_min_x - point_x)
        if point_y < obs_min_y: #below
            return (obs_min_y - point_y) + (obs_min_x - point_x)
        return obs_min_x - point_x
    if point_x > obs_max_x: #to the right
        if point_y > obs_max_y: #above
            return (point_y - obs_max_y) + (point_x - obs_max_x)
        if point_y < obs_min_y: #below
            return (obs_min_y - point_y) + (point_x - obs_max_x)
        return  point_x - obs_max_x
    return 0 #should only get here if point within shape bounds


def main():

    fnames = '../h5 files/box2.h5'
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, 1)
    data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    data = data - data[-1, :]
    data = data * 100
    data[:, 0] = -1 * abs(data[:, 0])
    data[:, 1] = abs(data[:, 1])
    traj = DouglasPeuckerPoints(data, 100)
    #print(traj)
    
    bx = box(-48.5, 12.5, -31.5, 19.5)
    
    switch_ind = 50
    a = 0.001
    b = 0.0001
    
    endpoints = [(-10, 0), (0, 5), (5, 2), (2, -5)]

    PA = ELTE_Perturbation_Analysis(traj, stretch=a, bend=b)
    x_prob = PA.setup_problem()
    constraints = [cp.abs(x_prob[0] - traj[0, 0]) <= 0, cp.abs(x_prob[PA.n_pts] - traj[0, 1]) <= 0, cp.abs(x_prob[PA.n_pts-1] - traj[-1, 0]) <= 0, cp.abs(x_prob[-1] - traj[-1, 1]) <= 0]
    for i in range(len(traj)):
        manh = get_min_manhattan([-48.5, 12.5, -31.5, 19.5], traj[i])
        print(manh)
        constraints.append(cp.abs(x_prob[i] - traj[i, 0]) + cp.abs(x_prob[PA.n_pts+i] - traj[i, 1]) - manh <= 0)
    sol = PA.solve_problem(constraints)
    
    cur_traj = sol[:switch_ind]
    
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], 'k', lw=3, label='Demonstration')
    plt.plot(cur_traj[:, 0], cur_traj[:, 1], 'r', lw=3, label='Current Reproduction')
    
    for endpoint in endpoints:
        print(endpoint)
        PA_cur = ELTE_Traj_Continuation(traj, cur_traj, stretch=a, bend=b)
        x_prob, cur_x_prob = PA_cur.setup_problem()
        cur_constraints = [cp.abs(x_prob[PA_cur.n_pts-PA_cur.cur_ind-1] - endpoint[0]) <= 0, cp.abs(x_prob[-1] - endpoint[1]) <= 0]
        for i in range(switch_ind, len(traj)):
            manh = get_min_manhattan([-48.5, 12.5, -31.5, 19.5], traj[i])
            print(manh)
            cur_constraints.append(cp.abs(x_prob[i-switch_ind] - traj[i, 0]) + cp.abs(x_prob[(PA_cur.n_pts-PA_cur.cur_ind)+(i-switch_ind)] - traj[i, 1]) - manh <= 0)
        sol, full_sol = PA_cur.solve_problem(cur_constraints)
        print(sol[PA_cur.n_pts-PA_cur.cur_ind-1])
        repro, = plt.plot(full_sol[:, 0], full_sol[:, 1], 'r--', lw=3)
        plt.plot(full_sol[-1, 0], full_sol[-1, 1], 'bx', ms=12, mew=3)
        #PA_cur.plot_solved_problem()
        #plt.show()
    
    repro, = plt.plot(full_sol[:, 0], full_sol[:, 1], 'r--', lw=3, label='Continuations')
    plt.plot(full_sol[-1, 0], full_sol[-1, 1], 'bx', ms=12, mew=3, label='Endpoints')
    plt.plot(traj[-1, 0], traj[-1, 1], 'kx', ms=12, mew=3, label='OG Endpoint')
    init, = plt.plot(full_sol[0, 0], full_sol[0, 1], 'ko', ms=12, label='Initial Point')
    init, = plt.plot(sol[0, 0], sol[0, 1], 'k*', ms=12, label='Current Point')
    
    plot_polygon(bx, ax=plt.gca(), add_points=False, color='g', alpha=0.5, label='Obstacle')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
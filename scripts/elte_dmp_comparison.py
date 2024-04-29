import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from shapely import box
from shapely.plotting import plot_polygon, plot_line

from elte import *
from utils import *
from downsampling import *

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './dmp_pastor_2009/')
import perform_dmp as dmp

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

    fnames = '../h5 files/pear.h5'
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, 1)
    data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    data = data - data[-1, :]
    data = data * 100
    data[:, 0] = -1 * abs(data[:, 0])
    data[:, 1] = abs(data[:, 1])
    traj = DouglasPeuckerPoints(data, 100)
    #print(traj)
    
    
    switch_ind = 30
    a = 0.00001
    b = 0.0001
    
    endpoints = [np.array([7, 2])]

    PA = ELTE_Perturbation_Analysis(traj, stretch=a, bend=b)
    x_prob = PA.setup_problem()
    constraints = [cp.abs(x_prob[0] - traj[0, 0]) <= 0, cp.abs(x_prob[PA.n_pts] - traj[0, 1]) <= 0, cp.abs(x_prob[PA.n_pts-1] - traj[-1, 0]) <= 0, cp.abs(x_prob[-1] - traj[-1, 1]) <= 0]
    sol = PA.solve_problem(constraints)
    
    cur_traj = sol[:switch_ind]
    
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], 'k', lw=3, label='Demonstration')
    
    K = 120
    D = 40
    
    dmp_x = dmp.perform_dmp_perturbed(traj[:, 0], initial=traj[0, 0], end=traj[-1, 0], ind=switch_ind, new_end=traj[-1, 0] + endpoints[0][0], k=K, d=D)
    dmp_y = dmp.perform_dmp_perturbed(traj[:, 1], initial=traj[0, 1], end=traj[-1, 1], ind=switch_ind, new_end=traj[-1, 1] + endpoints[0][1], k=K, d=D)
    
    dmp_x0 = dmp.perform_dmp_perturbed(traj[:, 0], initial=traj[0, 0], end=traj[-1, 0], k=K, d=D)
    dmp_y0 = dmp.perform_dmp_perturbed(traj[:, 1], initial=traj[0, 1], end=traj[-1, 1], k=K, d=D)
    #dmp_repro, = plt.plot(dmp_x0, dmp_y0, 'm', lw=3)
    
    dmp_repro, = plt.plot(dmp_x[0:switch_ind+1], dmp_y[0:switch_ind+1], 'm', lw=3, label='DMP')
    dmp_repro, = plt.plot(dmp_x[switch_ind:], dmp_y[switch_ind:], 'm--', lw=3, label='DMP Continuation')
    
    plt.plot(cur_traj[:, 0], cur_traj[:, 1], 'r', lw=3, label='ELTE')
    
    for endpoint in endpoints:
        PA_cur = ELTE_Traj_Continuation(traj, cur_traj, stretch=a, bend=b)
        x_prob, cur_x_prob = PA_cur.setup_problem()
        cur_constraints = [cp.abs(x_prob[PA_cur.n_pts-PA_cur.cur_ind-1] - endpoint[0]) <= 0, cp.abs(x_prob[-1] - endpoint[1]) <= 0]
        sol, full_sol = PA_cur.solve_problem(cur_constraints)
        repro, = plt.plot(full_sol[:, 0], full_sol[:, 1], 'r--', lw=3)
        plt.plot(full_sol[-1, 0], full_sol[-1, 1], 'kx', ms=12, mew=3)
    
    
    repro, = plt.plot(full_sol[:, 0], full_sol[:, 1], 'r--', lw=3, label='ELTE Continuation')
    plt.plot(traj[-1, 0], traj[-1, 1], 'kx', ms=12, mew=3, label='Endpoint')
    plt.plot(full_sol[-1, 0], full_sol[-1, 1], 'bx', ms=12, mew=3, label='New Endpoint')
    init, = plt.plot(full_sol[0, 0], full_sol[0, 1], 'ko', ms=12, label='Initial Point')
    init, = plt.plot(sol[0, 0], sol[0, 1], 'k*', ms=12, label='Current Point')
    init, = plt.plot(dmp_x[switch_ind], dmp_y[switch_ind], 'k*', ms=12)
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
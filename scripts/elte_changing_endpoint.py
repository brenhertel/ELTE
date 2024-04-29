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
    N = 100
    fnames = '../h5 files/box2.h5'
    [[sm_t, sm_x, sm_y], [unsm_t, unsm_x, unsm_y], [norm_t, norm_x, norm_y]] = scr2.read_demo_h5(fnames, 2)
    data = np.hstack((np.reshape(norm_x, (len(norm_x), 1)), np.reshape(norm_y, (len(norm_y), 1))))
    data = data - data[-1, :]
    data = data * 100
    data[:, 0] = -1 * abs(data[:, 0])
    data[:, 1] = abs(data[:, 1])
    traj = DouglasPeuckerPoints(data, N)
    #print(traj)
    
    
    a = 0.00001
    b = 0.00001
    

    PA = ELTE_Perturbation_Analysis(traj, stretch=a, bend=b)
    x_prob = PA.setup_problem()
    constraints = [cp.abs(x_prob[0] - traj[0, 0]) <= 0, cp.abs(x_prob[PA.n_pts] - traj[0, 1]) <= 0, cp.abs(x_prob[PA.n_pts-1] - traj[-1, 0]) <= 0, cp.abs(x_prob[-1] - traj[-1, 1]) <= 0]
    sol = PA.solve_problem(constraints)
    
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], 'k', lw=3, label='Demonstration')
    plt.plot(sol[:, 0], sol[:, 1], 'r--', lw=3, label='Continuations')
    
    final, = plt.plot(sol[-1, 0], sol[-1, 1], 'kx', ms=12, mew=3, label='Unperturbed Endpoint')
    
    for i in range(30, N - 20, 15):
        dx = -0.5 * i * np.sin(0.005 * i)
        dy = -0.05 * (i * np.sin(0.005 * i))
        
        cur_traj = sol[:i]
        
        PA_cur = ELTE_Traj_Continuation(traj, cur_traj, stretch=a, bend=b)
        x_prob, cur_x_prob = PA_cur.setup_problem()
        cur_constraints = [cp.abs(x_prob[PA_cur.n_pts-PA_cur.cur_ind-1] - (traj[-1, 0]+dx)) <= 0, cp.abs(x_prob[-1] - (traj[-1, 1]+dy)) <= 0]
        cont_sol, full_sol = PA_cur.solve_problem(cur_constraints, disp=False)
        
        repro, = plt.plot(full_sol[:, 0], full_sol[:, 1], 'r--', lw=3)
        plt.plot(traj[-1, 0] + dx, traj[-1, 1] + dy, 'bx', ms=12, mew=3)
        
        sol = full_sol
    
    
    repro, = plt.plot(full_sol[:, 0], full_sol[:, 1], 'r', lw=3, label='Final Continuation')
    
    adapt_inds = [30, 45, 60, 75]
    for j in range(len(adapt_inds)):
        init, = plt.plot(full_sol[adapt_inds[j], 0], full_sol[adapt_inds[j], 1], 'k*', ms=12)
    init, = plt.plot(full_sol[adapt_inds[j], 0], full_sol[adapt_inds[j], 1], 'k*', ms=12, label='Current Points')
    
    plt.plot(traj[-1, 0] + dx, traj[-1, 1] + dy, 'bx', ms=12, mew=3, label='New Endpoints')
    init, = plt.plot(sol[0, 0], sol[0, 1], 'ko', ms=12, label='Inital Point')
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
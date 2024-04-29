import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from shapely import box
from shapely.plotting import plot_polygon, plot_line

from elte import *
from utils import *
from downsampling import *

import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 14})

def read_data_new(fname):
    hf = h5py.File(fname, 'r')
    print(list(hf.keys()))
    js = hf.get('joint_state_info')
    joint_pos = np.array(js.get('joint_positions'))
    joint_vel = np.array(js.get('joint_velocities'))
    joint_eff = np.array(js.get('joint_effort'))
    joint_data = [joint_pos, joint_vel, joint_eff]
    
    tf = hf.get('toolpose_info')
    tf_pos = np.array(tf.get('toolpose_positions'))
    tf_rot = np.array(tf.get('toolpose_orientations'))
    cart_data = [tf_pos, tf_rot]
    
    hf.close()
    
    return joint_data, cart_data

def main():
    N = 50
    fname = '../h5 files/recorded_demo 2023-05-22 14_20_52.h5'
    
    joint_data, cart_data = read_data_new(fname)
    
    traj = cart_data[0]
    
    ds_traj = DouglasPeuckerPoints(traj, N)
    
    a = 0.001
    b = 0.001
    
    x = ds_traj[-1, 0]
    y = ds_traj[-1, 1]
    z = ds_traj[0, 2] + 0.05
    
    PA = ELTE_Perturbation_Analysis(ds_traj, stretch=a, bend=b)
    x_prob = PA.setup_problem()
    constraints = [cp.abs(x_prob[0] - ds_traj[0, 0]) <= 0, 
                   cp.abs(x_prob[PA.n_pts] - ds_traj[0, 1]) <= 0, 
                   cp.abs(x_prob[PA.n_pts2] - ds_traj[0, 2]) <= 0, 
                   cp.abs(x_prob[PA.n_pts-1] - x) <= 0, 
                   cp.abs(x_prob[PA.n_pts2-1] - y) <= 0, 
                   cp.abs(x_prob[-1] - z) <= 0]
    repro = PA.solve_problem(constraints)
    full_repro = repro
    
    #NOTE: These constraints are estimates for the box movement, the actual motion and timing was not recorded.
    for i in range(1, N):
        x = x + ((N - i) / 10000.)
        cur_traj = full_repro[0:i]
        PA_cur = ELTE_Traj_Continuation(ds_traj, np.array(cur_traj), stretch=a, bend=b)
        x_prob, cur_x_prob = PA_cur.setup_problem()
        cur_constraints = [cp.abs(cur_x_prob[PA_cur.n_pts-1] - x) <= 0, 
                           cp.abs(cur_x_prob[PA_cur.n_pts2-1] - y) <= 0, 
                           cp.abs(cur_x_prob[-1] - z) <= 0]
        repro, full_repro = PA_cur.solve_problem(cur_constraints, disp=False)
        
    
    
    plt.rcParams['figure.figsize'] = (9, 7)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'k', lw=3, label='Demonstration')
    ax.plot(full_repro[:, 0], full_repro[:, 1], full_repro[:, 2], 'r', lw=3, label='Adapted ELTE')
    
    final, = ax.plot(traj[0, 0], traj[0, 1], traj[0, 2], 'k.', ms=12, mew=3, label='Initial Point')
    final, = ax.plot(traj[-1, 0], traj[-1, 1], traj[-1, 2], 'kx', ms=12, mew=3, label='Endpoint')
    final, = ax.plot(x, y, z, 'bx', ms=12, mew=3, label='New Endpoint')
    
    plot_cube2(x - 0.02, x + 0.06, y - 0.05, y - 0.1, z - 0.07, z + 0.05, ax, color='green')
    plot_cube2(x - 0.2, x - 0.12, y - 0.05, y - 0.1, z - 0.07, z + 0.05, ax, color='green', alpha=0.2)
    
    import matplotlib.patches as mpatches
    color_patch = mpatches.Patch(color='green', label='legend')
    
    ax.axis('equal')
    ax.legend(loc='upper left')
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_zticklabels([])
    ax.view_init(azim=131, elev=51)
    
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()
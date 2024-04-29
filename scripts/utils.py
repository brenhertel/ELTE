import h5py
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib
import matplotlib.pyplot as plt

def mysavefig(figh, name):
    figh.savefig(name + ".png", bbox_inches='tight', dpi=300)
    matplotlib.use('pgf')
    plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
    figh.savefig(name + ".pgf")
    plt.close(figh)

def get_lasa_trajN(shape_name, n=1):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    filename = '../h5 files/lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo' + str(n))
    pos_info = demo.get('pos')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    #close out file
    hf.close()
    return [x_data, y_data]
    
def get_lasa_velN(shape_name, n=1):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    filename = '../h5 files/lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo' + str(n))
    pos_info = demo.get('vel')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    #close out file
    hf.close()
    return [x_data, y_data]
    
def read_3D_h5(fname):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    hf = h5py.File(fname, 'r')
    #navigate to necessary data and store in numpy arrays
    demo = hf.get('demo1')
    tf_info = demo.get('tf_info')
    pos_info = tf_info.get('pos_rot_data')
    pos_data = np.array(pos_info)
    
    x = pos_data[0, :]
    y = pos_data[1, :]
    z = pos_data[2, :]
    #close out file
    hf.close()
    return [x, y, z]

def read_jaco_data(fname):
    hf = h5py.File(fname, 'r')
    print(list(hf.keys()))
    js = hf.get('joint_state_info')
    joint_pos = np.array(js.get('joint_positions'))
    joint_vel = np.array(js.get('joint_velocities'))
    joint_eff = np.array(js.get('joint_effort'))
    joint_data = [joint_pos, joint_vel, joint_eff]
    
    tf = hf.get('cartesian_info')
    tf_pos = np.array(tf.get('positions'))
    tf_rot = np.array(tf.get('orientations'))
    cart_data = [tf_pos, tf_rot]
    #print(cart_data)
    
    
    gp = hf.get('torque_info')
    gripper_pos = np.array(gp.get('torque'))
    torq_data = gripper_pos
    
    hf.close()
    
    return joint_data, cart_data, torq_data
    
def display_data(joint_data, cart_data, torq_data):
    print('joint_positions: ' + str(np.shape(joint_data[0])))
    print('joint_velocities: ' + str(np.shape(joint_data[1])))
    print('joint_effort: ' + str(np.shape(joint_data[2])))
    
    print('positions: ' + str(np.shape(cart_data[0])))
    print('orientations: ' + str(np.shape(cart_data[1])))
    
    
    print('torque: ' + str(np.shape(torq_data)))
    return

def curvature_comparison(exp_data, num_data):
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    L = 2.*np.diag(np.ones((n_points,))) - np.diag(np.ones((n_points-1,)),1) - np.diag(np.ones((n_points-1,)),-1)
    L[0,1] = -2.
    L[-1,-2] = -2.
    err_abs = np.absolute(np.subtract(np.matmul(L, exp_data), np.matmul(L, num_data)))
    return np.sum(err_abs)

def herons_formula(a, b, c):
    s = (a + b + c) / 2.
    return (s * (s - a) * (s - b) * (s - c))**0.5

def swept_error_area(exp_data, num_data):
    #naive approach
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points - 1):
        p1 = exp_data[i]
        p2 = exp_data[i + 1]
        p3 = num_data[i + 1]
        p4 = num_data[i]
        p1_p2_dist = np.linalg.norm(p1 - p2)
        p1_p3_dist = np.linalg.norm(p1 - p3)
        p1_p4_dist = np.linalg.norm(p1 - p4)
        p2_p3_dist = np.linalg.norm(p2 - p3)
        p3_p4_dist = np.linalg.norm(p3 - p4)
        triangle1_area = herons_formula(p1_p4_dist, p1_p3_dist, p3_p4_dist)
        triangle2_area = herons_formula(p1_p2_dist, p1_p3_dist, p2_p3_dist)
        sum += triangle1_area + triangle2_area
    return sum / n_points

def sum_of_squared_error(exp_data, num_data):
    #naive approach
    (n_points, n_dims) = np.shape(exp_data)
    if not np.shape(exp_data) == np.shape(num_data):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points):
        sum += (np.linalg.norm(exp_data[i] - num_data[i]))**2
    return sum
    
def calc_jerk(traj):
    (n_pts, n_dims) = np.shape(traj)
    ttl = 0.
    for i in range(2, n_pts - 2):
        ttl += np.linalg.norm(traj[i - 2] + 2*traj[i - 1] - 2 * traj[i+1] - traj[i+2])
    return ttl
    
def align_ang_sim(exp_data, num_data):
    #print(np.shape(exp_data))
    #print(np.shape(num_data))
    exp = exp_data - exp_data[0]
    num = num_data - num_data[0]
    #assume exp data is always larger
    new_exp = [exp[0]]
    indeces = [i for i in range(1, len(exp) - 1)]
    for i in range(1, len(num) - 1):
        best_ind = indeces[0]
        best_val = np.linalg.norm(exp[indeces[0]] - num[i])
        for j in indeces:
            val = np.linalg.norm(exp[j] - num[i])
            if val < best_val:
                best_val = val
                best_ind = j
        #print([best_ind, exp[best_ind]])
        new_exp.append(exp[best_ind])
        indeces.remove(best_ind)
    new_exp.append(exp[-1])
    new_exp = np.array(new_exp)
    #plt.figure()
    #plt.plot(exp[:, 0], exp[:, 1], 'r.')
    #plt.plot(new_exp[:, 0], new_exp[:, 1], 'g.')
    #plt.plot(num[:, 0], num[:, 1], 'b.')
    #plt.show()
    return angular_similarity(new_exp, num)    

def angular_similarity(exp_data, num_data):
    (n_points, n_dims) = np.shape(exp_data)
    #print(np.shape(exp_data))
    #print(np.shape(num_data))
    if not (np.shape(exp_data) == np.shape(num_data)):
        print('Array dims must match!')
    sum = 0.
    for i in range(n_points - 1):
        v1 = exp_data[i + 1] - exp_data[i]
        v2 = num_data[i + 1] - num_data[i]
        #calc cosine similarity
        costheta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #calc angular distance
        ang_dist = np.arccos(costheta) / np.pi if not np.isnan(costheta) else 0
        sum = sum + ang_dist
    return sum / n_points
    
def plot_cube(min_x, max_x, min_y, max_y, min_z, max_z, ax):    
    #squares in x-y plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [min_z, min_z, min_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [max_z, max_z, max_z, max_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #squares in x-z plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, min_y, min_y, min_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [max_y, max_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #squares in y-z plane
    xv = [min_x, min_x, min_x, min_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [max_x, max_x, max_x, max_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    return
    
    
def plot_cube2(min_x, max_x, min_y, max_y, min_z, max_z, ax, color='saddlebrown', alpha=0.3):    
    #squares in x-y plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [min_z, min_z, min_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [max_z, max_z, max_z, max_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    
    #squares in x-z plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, min_y, min_y, min_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [max_y, max_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    
    #squares in y-z plane
    xv = [min_x, min_x, min_x, min_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    xv = [max_x, max_x, max_x, max_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor(color)
    poly.set_alpha(alpha)
    ax.add_collection3d(poly)
    return
    
def plot_irregular_cube(v1, v2, v3, v4, v5, v6, v7, v8, ax):    
    #front and back
    xv = [v1[0], v2[0], v3[0], v4[0]]
    yv = [v1[1], v2[1], v3[1], v4[1]]
    zv = [v1[2], v2[2], v3[2], v4[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [v5[0], v6[0], v7[0], v8[0]]
    yv = [v5[1], v6[1], v7[1], v8[1]]
    zv = [v5[2], v6[2], v7[2], v8[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #sides
    xv = [v1[0], v2[0], v6[0], v5[0]]
    yv = [v1[1], v2[1], v6[1], v5[1]]
    zv = [v1[2], v2[2], v6[2], v5[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [v3[0], v4[0], v8[0], v7[0]]
    yv = [v3[1], v4[1], v8[1], v7[1]]
    zv = [v3[2], v4[2], v8[2], v7[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    
    #sides
    xv = [v1[0], v5[0], v8[0], v4[0]]
    yv = [v1[1], v5[1], v8[1], v4[1]]
    zv = [v1[2], v5[2], v8[2], v4[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    xv = [v2[0], v6[0], v7[0], v3[0]]
    yv = [v2[1], v6[1], v7[1], v3[1]]
    zv = [v2[2], v6[2], v7[2], v3[2]]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_facecolor('saddlebrown')
    poly.set_alpha(0.3)
    ax.add_collection3d(poly)
    return
    
#this function from https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid
    
def plot_3D_cylinder(ax, radius, height, elevation=0, x_center = 0, y_center = 0, resolution=100, color='r'):
    #fig=plt.figure()
    #ax = Axes3D(fig, azim=30, elev=30)

    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem


    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")
    return
    
# Plotting functions

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm
import math
import numpy as np

def plot_hvect_count(hvect_counter, max):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_0$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_1$", fontsize=20, labelpad=15)

    for hvect0, c0 in hvect_counter.items():
        p0 = hvect0
        if c0==0:
            ax.scatter( p0[0], p0[1],  marker='s', c = 'grey', s=100)
        else:
            dotsize = math.sqrt(c0/max)*8000
            ax.scatter( p0[0], p0[1],  marker='.', c = 'black', s=dotsize+50)


def _getBoundedWedge(angle0, angle1):
    #Returns a matplotlib mpath wedge with a phantom bounding box
    wedge0= mpath.Path.wedge(angle0, angle1)
    #print(f"wedge0:{wedge0}")

    vertices = wedge0.vertices
    codes = wedge0.codes

    #Add ghost vertices to define bounding box
    vertices= np.insert( vertices, 0, [[1.1,1.1], [-1.1,1.1] , [-1.1,-1.1], [1.1,-1.1]] , axis=0)
    codes = np.insert( codes, 0, [1,1,1,1])

    wedgeextra = mpath.Path(vertices, codes)
    
    return wedgeextra

def _getPieSlicesPointedToMaxEdge_2class(data, hvector):

        #Project hvector to xyplane to get angle
        #Returns a list of paths to be used to draw a pie chart
        #No color is defined here

        #Sum data
        data_np = np.array(data)
        data_sum = data_np.sum()

        #2 class means that there are only one possible angles class zero wedge
        angles_for_class_zero = -45

        if data_sum>0:
            #print(f"hvector={hvector} , data={data}")
            #calculate x coordinates of each slice
            angle_widths = data_np/data_sum * 360 #angles in degrees for path methods
            
            end_angle_positions = np.cumsum(angle_widths)
            #print(f"end_angle_positions={end_angle_positions}")
            
            #Get the angle to start
            #data_argmax = np.argmax(data_np) #gets the major class reference number
            anglewidth_of_class_zero = angle_widths[0]
            anglepos_of_class_zero_midpoint = end_angle_positions[0]-anglewidth_of_class_zero/2

            angletarget = angles_for_class_zero

            angle_to_rotate_pies = angletarget - anglepos_of_class_zero_midpoint 

            angle_allpositions = np.append(np.array([0]), end_angle_positions)
            angle_positions_rotated = angle_allpositions + angle_to_rotate_pies

            wedges = []
            for i in range(len(angle_positions_rotated)-1):
                angle0= angle_positions_rotated[i]
                angle1= angle_positions_rotated[i+1]
                
                dangle = angle1-angle0

                wedge0=None
                if dangle>0.1:
                    if dangle >=360:
                        wedge0 = mpath.Path.unit_circle()
                    else:
                        #wedge0= mpath.Path.wedge(angle0, angle1)
                        wedge0 = _getBoundedWedge(angle0,angle1)

                wedges.append(wedge0)

            return wedges
        else:
            return None


def plot_hvectors_w_classCountAsPies_oriented_viridis(hvect_gndclass_counter_data1):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_{0}$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_{1}$", fontsize=20, labelpad=15)
    
    mycmap = matplotlib.cm.get_cmap('viridis')

    for hvect, d0 in hvect_gndclass_counter_data1.items():
        wedges = _getPieSlicesPointedToMaxEdge_2class(d0, hvect)
        p0 = hvect

        if wedges is None:
            ax.scatter( p0[0], p0[1], marker='s', c = 'grey', s=50)
        else:
            for i, w0 in enumerate(wedges):
                if not w0 is None:
                    c0 = float(i)/2.0
                    ax.scatter( p0[0], p0[1], marker=w0, c=[mycmap(c0)], s=500)

def plotHvectToClassAssignment_viridis(hvect_to_class_dict):
    # Plot the surface.
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_{0}$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_{1}$", fontsize=20, labelpad=15)

    mycmap = matplotlib.cm.get_cmap('viridis')

    #points need to be passed to the scatter plot one by one
    for h0, c0 in hvect_to_class_dict.items():
        c1 = float(c0)/2.0
        ax.scatter(h0[0], h0[1],  marker = '.',c=[mycmap(c1)],s=1000)
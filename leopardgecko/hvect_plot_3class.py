import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.cm
import math
import numpy as np

# plotting for 3 class

def plot_hvect_count_3class(hvect_counter, max):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,45)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_0$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_1$", fontsize=20, labelpad=15)
    ax.set_zlabel("$n_2$", fontsize=20, labelpad=15)
    
    segmcolors=('black','red','green','blue','pink')

    for hvect0, c0 in hvect_counter.items():
        p0 = hvect0
        if c0==0:
            ax.scatter( p0[0], p0[1], p0[2], marker='s', c = 'grey', s=100)
        else:
            dotsize = math.sqrt(c0/max)*8000
            ax.scatter( p0[0], p0[1], p0[2], marker='.', c = 'black', s=dotsize+50)

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


def plot_hvectors_w_classCountAsPies(hvect_gndclass_counter_data1):

    def getPieSlices(data):
        #Returns a list of paths to be used to draw a pie chart
        #No color is defined here

        #Sum data
        data_np = np.array(data)
        data_sum = data_np.sum()

        if data_sum>0:
            #creates a wedge for each, with same centre point
            centre=[0,0,0]
            #calculate x coordinates of each slice
            angle_widths = data_np/data_sum * 360 #angles in degrees for path methods
            
            wedges = []
            
            angle0=90
            for aw in angle_widths:
                wedge0=None
                if aw>0:
                    angle1 = angle0-aw
                    #get the wedge
                    wedge0= mpath.Path.wedge(angle1, angle0)
                    angle0=angle1
                wedges.append(wedge0)

            return wedges
        else:
            return None
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,45)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("n0", fontsize=20, labelpad=15)
    ax.set_ylabel("n1", fontsize=20, labelpad=15)
    ax.set_zlabel("n2", fontsize=20, labelpad=15)

    segmcolors=('black','red','green','blue','pink')

    for hvect, d0 in hvect_gndclass_counter_data1.items():
        wedges = getPieSlices(d0)
        p0 = hvect

        if wedges is None:
            ax.scatter( p0[0], p0[1], p0[2], marker='s', c = 'grey', s=50)
        else:
            for i, w0 in enumerate(wedges):
                if not w0 is None:
                    ax.scatter( p0[0], p0[1], p0[2], marker=w0, c = segmcolors[i], s=500)


class cQuaternion:
    def __init__(self,w,x,y,z):
        self.w=w
        self.x=x
        self.y=y
        self.z=z

    def normalise(self):
        #Normalises the quaternion
        q0 = np.array( [self.w,self.x, self.y,self.z ] )
        norm = np.linalg.norm(q0)
        assert norm!=0, "Error, quaternion is zero. Cannot normalise"
        q1 = q0 / norm
        q2= cQuaternion(q1[0],q1[1],q1[2],q1[3])
        return q2
        
    @staticmethod
    def fromVectorAndAngle(vectorXYZ, theta_rad):
        if len(vectorXYZ)==3:
            w= math.cos(theta_rad/2)

            vnp = np.array(vectorXYZ)
            norm =  np.linalg.norm(vnp)
            assert norm!=0 , "Cannot use a vector that has zero length"

            vnp_norm = vnp/norm
            sin2 = math.sin(theta_rad/2)
            x = sin2*vnp_norm[0]
            y = sin2*vnp_norm[1]
            z = sin2*vnp_norm[2]

            #print(f"w:{w}, x:{x}, y:{y}, z:{z}")
            qres = cQuaternion(w,x,y,z)
            return qres
        else:
            return None
    
    def __mul__(self, q2):
        #Hamilton product
        #https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
        
        q1 = self

        w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
        x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
        y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
        z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w

        res = cQuaternion(w,x,y,z)
        return res
    
    def __add__(self,q2):
        q1=self
        w = q1.w + q2.w
        x = q1.x + q2.x
        y = q1.y + q2.y
        z = q1.z + q2.z
        res = cQuaternion(w,x,y,z)
        return res
    
    def __sub__(self,q2):
        q1=self
        w = q1.w - q2.w
        x = q1.x - q2.x
        y = q1.y - q2.y
        z = q1.z - q2.z
        res = cQuaternion(w,x,y,z)
        return res

    def getInverse(self):
        #For the inverse we just need to negate the imaginary (pseudo vector) parts
        
        return cQuaternion(self.w, -self.x,-self.y,-self.z)

    def transformVector(self,vectorXYZ):
        #Use this quaternion to transform vector given
        #It is similar to quaternion*quaternion, however with the first quaternion having w=0

        qv = cQuaternion(0, vectorXYZ[0], vectorXYZ[1], vectorXYZ[2]) #Careful to not normalise
        qtemp = qv * self.getInverse()

        qres = self*qtemp

        #print(f"qres = {qres}")
        #assert qres.w==0, print(f"Error in rotation transformation, final qres.w={qres.w} is not zero.")

        return [qres.x,qres.y,qres.z]
    
    def __str__(self):
        vres,theta = self.getVectorAndRotationAngle()

        return (f"w={self.w}, x={self.x}, y={self.y}, z={self.z}, rotaxis ={vres}, theta={theta}")

    def getVectorAndRotationAngle(self):
        v3 = np.array([self.z, self.y, self.x])

        v3_norm = np.linalg.norm(v3)
        vres = (v3/v3_norm).tolist()
        theta = 2*math.atan2(v3_norm, self.w)

        return vres, theta


qRot0 = cQuaternion.fromVectorAndAngle([0,0,1], - math.pi*3/4 ) #Rotate around zz
qRot1 = cQuaternion.fromVectorAndAngle([1,0,0], -math.atan( math.sqrt(2))) #Rotate around xx

qRot2 = qRot1*qRot0

transl0 = np.array([0,-6,-6])

def _transformHvectorToXY(hvector):
    hv_np = np.array(hvector)
    hv_sum = np.sum(hv_np)

    transl0 = np.array([-hv_sum/2,-hv_sum/2,0])
    hv_transl = hv_np+transl0

    #print(f"hv_transl:{hv_transl}")
    hv_transf = qRot2.transformVector(hv_transl) #Warning! it assumes the first index is zz
    return hv_transf



def _getPieSlicesPointedToMaxEdge(data, hvector):

        #Project hvector to xyplane to get angle
        #Returns a list of paths to be used to draw a pie chart
        #No color is defined here

        #Sum data
        data_np = np.array(data)
        data_sum = data_np.sum()


        if data_sum>0:
            #print(f"hvector={hvector} , data={data}")
            #calculate x coordinates of each slice
            angle_widths = data_np/data_sum * 360 #angles in degrees for path methods
            
            end_angle_positions = np.cumsum(angle_widths)
            #print(f"end_angle_positions={end_angle_positions}")
            
            #Get the angle to start
            data_argmax = np.argmax(data_np)
            anglewidth_of_maxwedge = angle_widths[data_argmax]
            anglepos_of_maxwedge_midpoint = end_angle_positions[data_argmax]-anglewidth_of_maxwedge/2
            #print(f"anglepos_of_maxwedge_midpoint={anglepos_of_maxwedge_midpoint}")


            hv_np = np.array(hvector)
            ndirs = np.sum(hv_np)
            #print(f"ndirs={ndirs}")

            edgemax_pos = np.zeros(3)
            edgemax_pos[data_argmax]=ndirs

            #Now try to calculate angle after projecting hvector and edgepos
            #print(f"edgemax_pos={edgemax_pos}")
            edgepos_proj = np.array(_transformHvectorToXY(edgemax_pos))
            #print(f"hv_np={hv_np}")
            hv_proj = np.array(_transformHvectorToXY(hv_np))

            tmax_vect = edgepos_proj - hv_proj
            
            if np.linalg.norm(tmax_vect)<=1e-6:
                #print("Overlap with edge")
                foo = [ndirs/3, ndirs/3, ndirs/3]
                foo_proj = np.array(_transformHvectorToXY(foo))
                tmax_vect = edgepos_proj - foo_proj
            #print(f"tmax_vect={tmax_vect}, hopefully only x,y")

            theta = math.atan2(tmax_vect[1], tmax_vect[0]) # atan2(y,x)
            
            #Make sure that z component is zero
            #assert tmax_proj[0]==0, print("ERROR, z component of tmax_proj is not zero")

            angletarget = theta*180.0/math.pi
            # if angletarget<0:
            #     angletarget=360+angletarget
            #print(f"angletarget={angletarget}")

            angle_to_rotate_pies = angletarget - anglepos_of_maxwedge_midpoint 
            #angle_to_rotate_pies=0 #Test

            angle_allpositions = np.append(np.array([0]), end_angle_positions)
            angle_positions_rotated = angle_allpositions + angle_to_rotate_pies
            #angle_positions_rotated = angle_allpositions #Test, non-rotated
            #print(f"angle_positions_rotated:{angle_positions_rotated}")
        
            #print(f"hvector={hvector} , data={data}, angle_positions_rotated:{angle_positions_rotated}")

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
                #wedge0= mpath.Path.wedge(angle0, angle1)
                #print(f"wedge0:{wedge0}")
                wedges.append(wedge0)

            return wedges
        else:
            return None


tab10 = matplotlib.cm.get_cmap('tab10')

def plot_hvectors_w_classCountAsPies_oriented1(hvect_gndclass_counter_data1):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,45)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_{0}$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_{1}$", fontsize=20, labelpad=15)
    ax.set_zlabel("$n_{2}$", fontsize=20, labelpad=15)

    #segmcolors=('black','red','green')
    

    for hvect, d0 in hvect_gndclass_counter_data1.items():
        wedges = _getPieSlicesPointedToMaxEdge(d0, hvect)
        p0 = hvect

        if wedges is None:
            ax.scatter( p0[0], p0[1], p0[2], marker='s', c = 'grey', s=50)
        else:
            for i, w0 in enumerate(wedges):
                if not w0 is None:
                    #ax.scatter( p0[0], p0[1], p0[2], marker=w0, c = segmcolors[i], s=500)
                    #ax.scatter( p0[0], p0[1], p0[2], marker=w0, c = i, s=500, cmap="tab10")
                    ax.scatter( p0[0], p0[1], p0[2], marker=w0, c=[tab10(i)], s=500)

#OK. Wedges point to max edge, but marker wedges sizes (radius) are buggy

def plot_hvectors_w_classCountAsPies_oriented_viridis(hvect_gndclass_counter_data1):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,45)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_{0}$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_{1}$", fontsize=20, labelpad=15)
    ax.set_zlabel("$n_{2}$", fontsize=20, labelpad=15)
    
    mycmap = matplotlib.cm.get_cmap('viridis')

    for hvect, d0 in hvect_gndclass_counter_data1.items():
        wedges = _getPieSlicesPointedToMaxEdge(d0, hvect)
        p0 = hvect

        if wedges is None:
            ax.scatter( p0[0], p0[1], p0[2], marker='s', c = 'grey', s=50)
        else:
            for i, w0 in enumerate(wedges):
                if not w0 is None:
                    c0 = float(i)/2.0
                    ax.scatter( p0[0], p0[1], p0[2], marker=w0, c=[mycmap(c0)], s=500)


def plotHvectToClassAssignment_viridis(hvect_to_class_dict):
    # Plot the surface.
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,45)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_{0}$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_{1}$", fontsize=20, labelpad=15)
    ax.set_zlabel("$n_{2}$", fontsize=20, labelpad=15)

    mycmap = matplotlib.cm.get_cmap('viridis')

    #points need to be passed to the scatter plot one by one
    for h0, c0 in hvect_to_class_dict.items():
        c1 = float(c0)/2.0
        if c1>=0:
            ax.scatter(h0[0], h0[1], h0[2], marker = '.',c=[mycmap(c1)],s=1000)
        else: #In case of no assignment the value of the class is below zero
            ax.scatter( h0[0], h0[1], h0[2], marker='s', c = 'grey', s=100)


def plot_hvectors_w_classCountAsPies_oriented_viridis(hvect_gndclass_counter_data1):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45,45)
    ax.tick_params(labelsize=20)
    ax.set_xlabel("$n_{0}$", fontsize=20, labelpad=15)
    ax.set_ylabel("$n_{1}$", fontsize=20, labelpad=15)
    ax.set_zlabel("$n_{2}$", fontsize=20, labelpad=15)
    
    mycmap = matplotlib.cm.get_cmap('viridis')

    for hvect, d0 in hvect_gndclass_counter_data1.items():
        wedges = _getPieSlicesPointedToMaxEdge(d0, hvect)
        p0 = hvect

        if wedges is None:
            ax.scatter( p0[0], p0[1], p0[2], marker='s', c = 'grey', s=50)
        else:
            for i, w0 in enumerate(wedges):
                if not w0 is None:
                    c0 = float(i)/2.0
                    ax.scatter( p0[0], p0[1], p0[2], marker=w0, c=[mycmap(c0)], s=500)
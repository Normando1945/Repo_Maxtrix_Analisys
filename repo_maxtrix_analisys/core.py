import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


#########################################################################################################################################
#########################################################################################################################################
######################################################## Simple Matriz stack ############################################################
#########################################################################################################################################
#########################################################################################################################################
class SimpleMatrixStack:
    
    def __init__(self, hola):
        self.hola = hola
        
    def matriz(self):
        hola = self.hola
        
        auto = np.zeros([4,4])
        i = 0
        for j in np.arange(1,4+1,1):
            auto[i,:] = np.linspace((j * hola),(j * hola) +12,4)
            i = i+1
    
        return auto

#########################################################################################################################################
#########################################################################################################################################
##################################### Stiffness Matrix RMF (due Flexural + Shear deformation) ###########################################
#########################################################################################################################################
#########################################################################################################################################
class StiffnessMatrix_simple:
    '''
    hola
    '''
    
    def __init__(self, A,I,E,L,G,f,da = 0, db = 0):
        self.A = A
        self.I = I
        self.E = E
        self.L = L
        self.G = G
        self.f = f
        self.da = da
        self.db = db
    
    def stiffness_matrix_Element_RMF_L(self):
        '''
        Calculate the stiffness matrix inluding shear deformation of a RMF width 6 DOF
        '''
        A = self.A
        I = self.I
        E = self.E
        L = self.L
        G = self.G
        f = self.f
        
        Be = 6*E*I*f / (G*A*L**2)
        r = A*E /L
        kp = 2*E*I/L * (2 + Be)/(1 + 2*Be)
        ap = 2*E*I/L * (1 - Be)/(1 + 2*Be)
        bp = 6*E*I/(L**2) * (1)/(1 + 2*Be)
        tp = 12*E*I/(L**3) * (1)/(1 + 2*Be)

        kl = np.array([[r,0,0,-r,0,0],
                    [0,tp,bp,0,-tp,bp],
                    [0,bp,kp,0,-bp,ap],
                    [-r,0,0,r,0,0],
                    [0,-tp,-bp,0,tp,-bp],
                    [0,bp,ap,0,-bp,kp]], dtype=float)
    
        return kl
    
    def transform_matrix_df_to_indf(self):
        da = self.da
        db = self.db
        '''
        This function transform the stiffness matrix of deformable state to infeformable state
        '''
        Tr = np.array([[1,0,0,0,0,0],
                       [0,1,0,0,0,0],
                       [0,da,1,0,0,0],
                       [0,0,0,1,0,0],
                       [0,0,0,0,1,0],
                       [0,0,0,0,-db,1],
                       ])
        return Tr

#########################################################################################################################################
#########################################################################################################################################
############################ Stiffness Matrix for 2D MF Element with Rigid End Offsets and Shear Deformation ############################
#########################################################################################################################################
#########################################################################################################################################

class MF_K_T_L_Element2D:
    '''
    What this class does
    --------------------
    
    - Builds and returns the local stiffness matrix of the 2D MF element considering axial deformation, Timoshenko-type shear deformation, flexural    deformation, and the influence of rigid end offsets.
    '''
    def __init__(self, E, A, I, L, nu=0.20, f=6/5, dA=0.0, dB=0.0, thetha=0.0):                                     # Initialize MF element properties
        self.E  = E                                                                                                 # Modulus of elasticity
        self.A  = A                                                                                                 # Cross-sectional area
        self.I  = I                                                                                                 # Moment of inertia
        self.L  = L                                                                                                 # Length of the flexible span (between element end nodes)
        self.nu = nu                                                                                                # Poisson ratio (needed to compute shear modulus G)
        self.f  = f                                                                                                 # Shear correction factor (used in beta)
        self.dA = dA                                                                                                # Rigid end offset at end A (local)
        self.dB = dB                                                                                                # Rigid end offset at end B (local)
        self.thetha = thetha                                                                                        # Orientation angle in degrees

    def stiffness_matrix_MF_EI_AE_GAf_da_db(self):
        '''
        '''
        E  = self.E                                                                                                 # Using element properties (Modulus of elasticity)
        A  = self.A                                                                                                 # Using element properties (Cross-sectional area)
        I  = self.I                                                                                                 # Using element properties (Moment of inertia)
        L  = self.L                                                                                                 # Using element properties (Length of the flexible span)
        nu = self.nu                                                                                                # Using Poisson ratio
        f  = self.f                                                                                                 # Using shear correction factor
        dA = self.dA                                                                                                # Using rigid end offset at A
        dB = self.dB                                                                                                # Using rigid end offset at B

        # --- Axial stiffness term ---------------------------------------------------------------------------------
        r = A * E / L                                                                                               # r = AE/L

        # --- Shear deformation correction (Timoshenko-type) -------------------------------------------------------
        G = E / (1.0 + 2 * nu)                                                                                      # Shear modulus: G = E / [2(1+nu)]
        beta = 6*E*I*f / (G*A*L**2)                                                                                 # beta = (6EI)/(GA L^2) * f

        # --- Base (prime) coefficients including shear deformation ------------------------------------------------
        t1 = (12.0 * E * I) / (L**3) * (1.0 / (1.0 + 2.0 * beta))                                                   # t' = 12EI/L^3 * 1/(1+2beta)
        b1 = ( 6.0 * E * I) / (L**2) * (1.0 / (1.0 + 2.0 * beta))                                                   # b' =  6EI/L^2 * 1/(1+2beta)
        k1 = ( 2.0 * E * I) /  L      * ((2.0 + beta) / (1.0 + 2.0 * beta))                                         # k' =  2EI/L * (2+beta)/(1+2beta)
        a1 = ( 2.0 * E * I) /  L      * ((1.0 - beta) / (1.0 + 2.0 * beta))                                         # a' =  2EI/L * (1-beta)/(1+2beta)
        
        # --- Rigid end offsets correction (double/ triple prime coefficients) -------------------------------------
        b2 = b1 + t1 * dA                                                                                           # b''  = b' + t' dA
        b3 = b1 + t1 * dB                                                                                           # b''' = b' + t' dB
        k2 = k1 + 2.0 * b1 * dA + t1 * (dA**2)                                                                      # k''  = k' + 2 b' dA + t' dA^2

        # --- a'' and a''' are represented with the same expression-------------------------------------------------
        a2 = a1 + b1 * (dA + dB) + t1 * (dA * dB)                                                                   # a''  = a' + b'(dA+dB) + t' dA dB
        a3 = a2                                                                                                     # a''' = a'' (same expression per your sketch)

        k3 = k1 + 2.0 * b1 * dB + t1 * (dB**2)                                                                      # k''' = k' + 2 b' dB + t' dB^2

        # --- Local stiffness matrix for 2D MF element (axial + flexure + shear + rigid ends) ----------------------
        K_e = np.array([
            [  r,   0,    0,  -r,    0,    0 ],
            [  0,  t1,   b2,   0,  -t1,   b3 ],
            [  0,  b2,   k2,   0,  -b2,   a3 ],
            [ -r,   0,    0,   r,    0,    0 ],
            [  0, -t1,  -b2,   0,   t1,  -b3 ],
            [  0,  b3,   a2,   0,  -b3,   k3 ]
        ], dtype=float)

        return K_e                                                                                                  # Return the element stiffness matrix
    
    def transformation_matrix_2D(self):
        thetha_radian = self.thetha
        thetha = np.deg2rad(thetha_radian)                                                                          # Using orientation angle
        
        c = np.cos(thetha)                                                                                          # Cosine of the angle
        s = np.sin(thetha)                                                                                          # Sine of the angle

        T_MF = np.array([                                                                                           # Transformation matrix for 2D element
            [ c,  -s, 0,  0,  0, 0 ],
            [ s,   c, 0,  0,  0, 0 ],
            [ 0,   0, 1,  0,  0, 0 ],
            [ 0,   0, 0,  c, -s, 0 ],
            [ 0,   0, 0,  s,  c, 0 ],
            [ 0,   0, 0,  0,  0, 1 ]
        ], dtype=float)

        return T_MF                                                                                                 # Return the transformation matrix

#########################################################################################################################################
#########################################################################################################################################
############################################## Stiffness Matrix for TRUSS Element #######################################################
#########################################################################################################################################
#########################################################################################################################################

class ARM_K_T_Element2D:
    '''
    What this class does
    --------------------
    
    - Builds and returns the local stiffness matrix of the TRUSS element considering axial deformation.
    '''
    def __init__(self, E, A, L, thetha=0.0):                                                                        # Initialize TRUSS element properties
        self.E  = E                                                                                                 # Modulus of elasticity
        self.A  = A                                                                                                 # Cross-sectional area
        self.L  = L                                                                                                 # Length of the flexible span (between element end nodes)
        self.thetha = thetha                                                                                        # Orientation angle in degrees

    def stiffness_matrix_ARM_AE(self):
        '''
        '''
        E  = self.E                                                                                                 # Using element properties (Modulus of elasticity)
        A  = self.A                                                                                                 # Using element properties (Cross-sectional area)
        L  = self.L                                                                                                 # Using element properties (Length of the flexible span)
        
        # --- Axial stiffness term ---------------------------------------------------------------------------------
        r = A * E / L                                                                                               # r = AE/L

        # --- Local stiffness matrix for TRUSS element (axial) ----------------------
        K_ea = np.array([
                [r, -r],
                [-r, r]
                    ], dtype=float)

        return K_ea                                                                                                 # Return the element stiffness matrix
    
    def transformation_ARM_matrix_2D(self):
        thetha_radian = self.thetha
        thetha = np.deg2rad(thetha_radian)                                                                          # Using orientation angle
        
        c = np.cos(thetha)                                                                                          # Cosine of the angle
        s = np.sin(thetha)                                                                                          # Sine of the angle

        T_ARM = np.array([                                                                                          # Transformation matrix for 2D element
            [c, 0],
            [s, 0],
            [0, c],
            [0, s]
        ], dtype=float)

        return T_ARM                                                                                                # Return the transformation matrix


#########################################################################################################################################
#########################################################################################################################################
########################################### Assemble Stiffness Matrices of the Structure ################################################
#########################################################################################################################################
#########################################################################################################################################

class Assembler:
        def __init__(self, lee = any, K = any, S = any, nglt = any):
              self.lee = lee
              self.K = K
              self.S = S
              self.nglt = nglt
        
        def assambler_due_lee(self):
                lee = self.lee
                K = self.K
                S = self.S
                nglt = self.nglt
                
                ng = len(lee)
                for i in range(ng):
                        ii = int(lee[i])
                        if ii > 0 and ii <= nglt:
                                for j in range(ng):
                                        jj = int(lee[j])
                                        if jj > 0 and jj <= nglt:
                                                S[ii-1, jj-1] += K[i, j]
                return S

#########################################################################################################################################
#########################################################################################################################################
#################################################### Managing Multiple 2D MF Elements ###################################################
#########################################################################################################################################
#########################################################################################################################################

class Manager_K_T_elements2D:                                                                                       # Class for managing multiple 2D Matrix elements     
    def __init__(self, method = 'MF'):                                                                              # Initialize the list of elements
        self.elements = []                                                                                          # List to store Matrix elements
        self.method = method                                                                                        # Method to specify the type of element ('MF' for Matrix Frame, 'ARM' for Axial Rod/Truss)
        
    def add_element(self, element):                                                                                 # Method to add an element to the list                            
        self.elements.append(element)                                                                               # Append the element to the list                 

    def stacked_stiffness_matrices(self):                                                                           # Method to stack stiffness matrices of all elements         
        method = self.method                                                                                        # Get the method type to determine which stiffness matrix to use
        K_L_blocks = []                                                                                             # Initialize list to hold stiffness matrices                

        if method == 'MF':                                                                                          # If the method is 'MF', use the stiffness matrix that includes shear deformation and rigid end offsets
                for elem in self.elements:                                                                          # Loop through each element
                        K_L_blocks.append(elem.stiffness_matrix_MF_EI_AE_GAf_da_db())                               # Append the stiffness matrix of the element
        else:                                                                                                       # If the method is not 'MF' (assumed to be 'ARM'), use the simpler stiffness matrix for axial deformation only
                for elem in self.elements:                                                                          # Loop through each element
                        K_L_blocks.append(elem.stiffness_matrix_ARM_AE())                                           # Append the stiffness matrix of the element   
                
        return np.vstack(K_L_blocks)                                                                                # Return the stacked stiffness matrices as a single array
    
    def stacked_transformation_matrices(self):                                                                      # Method to stack transformation matrices of all elements         
        method = self.method                                                                                        # Get the method type to determine which transformation matrix to use   
        T_blocks = []                                                                                               # Initialize list to hold transformation matrices                

        if method == 'MF':                                                                                          # If the method is 'MF', use the transformation matrix for 2D MF elements
                for elem in self.elements:                                                                          # Loop through each element
                        T_blocks.append(elem.transformation_matrix_2D())                                            # Append the transformation matrix of the element
        else:                                                                                                       # If the method is not 'MF' (assumed to be 'ARM'), use the simpler transformation matrix for axial deformation only
                for elem in self.elements:                                                                          # Loop through each element
                        T_blocks.append(elem.transformation_ARM_matrix_2D())                                        # Append the transformation matrix of the element    
        
        return np.vstack(T_blocks)                                                                                  # Return the stacked transformation matrices as a single array
    


#########################################################################################################################################
#########################################################################################################################################
########################################################## Matrix Visualizer ############################################################
#########################################################################################################################################
#########################################################################################################################################

class M_visual_2D_3D:                                                                                               # Class for visualize any values of a matrix     
    def __init__(self, Matrix, color = 'seismic'):                                                                                     # Initialize the class with the matrix to be plotted
        self.ElemDraw = Matrix                                                                                      # Store the input matrix as an internal variable
        self.color = color
    def M_visual(self):                                                                                             # Method to generate the 2D and 3D visualization of the matrix values

        ElemDraw = self.ElemDraw                                                                                    # Local variable containing the matrix to be visualized
        color = self.color
        
        fig = plt.figure(figsize=(18, 9))                                                                           # Create the main figure with a wide format
        fig.suptitle("Representation Matrix Values",                                                                # Add a global title to the figure
             fontsize=18, fontweight='bold', color=(0, 0, 1))                                                       # Define font size, bold style, and blue color
        # -----------------------------------------------
        # Subplot 1: Matrix in 2D
        # -----------------------------------------------
        ax0 = fig.add_subplot(1, 2, 1)                                                                              # Create the first subplot for the 2D representation
        lim = np.max(np.abs(ElemDraw))                                                                              # Compute the maximum absolute value for symmetric color scaling
        im0 = ax0.imshow(ElemDraw, cmap= color, aspect='equal', vmin=-lim, vmax=lim)                                # Display the matrix as a 2D color map
        cbar0 = fig.colorbar(im0, ax=ax0, pad=0.01, fraction=0.03)                                                  # Add a colorbar associated with the 2D plot
        cbar0.set_label('Values')                                                                                   # Label the colorbar
        ax0.set_title('Matrix Values in 2D', fontweight='bold')                                                     # Set the title of the 2D subplot
        ax0.set_xlabel('GLDs')                                                                                      # Label the x-axis
        ax0.set_ylabel('GDLs')                                                                                      # Label the y-axis

        # -----------------------------------------------
        # Subplot 2: Same matrix in 3D
        # -----------------------------------------------
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')                                                             # Create the second subplot for the 3D bar representation
        nrows, ncols = ElemDraw.shape                                                                               # Get the number of rows and columns of the matrix
        xpos, ypos = np.meshgrid(np.arange(ncols), np.arange(nrows))                                                # Generate the grid positions for columns and rows
        xpos = xpos.ravel()                                                                                         # Flatten the x positions into a 1D array
        ypos = ypos.ravel()                                                                                         # Flatten the y positions into a 1D array
        zpos = np.zeros_like(xpos, dtype=float)                                                                     # Set the base height of all bars at z = 0
        dx = 0.8 * np.ones_like(zpos)                                                                               # Define the width of each bar in the x direction
        dy = 0.8 * np.ones_like(zpos)                                                                               # Define the width of each bar in the y direction
        dz = ElemDraw.ravel()                                                                                       # Flatten the matrix values to use them as bar heights
        norm = plt.Normalize(vmin=-lim, vmax=lim)                                                                   # Create a normalization object for consistent color scaling
        colors0 = plt.cm.seismic(norm(dz))                                                                          # Assign colors to each bar according to its value
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors0, shade=True)                                          # Plot the 3D bars with the corresponding colors
        ax1.set_xlim(0, ncols)                                                                                      # Set the limits of the x-axis
        ax1.set_ylim(0, nrows)                                                                                      # Set the limits of the y-axis
        ax1.set_zlim(-lim, lim)                                                                                     # Set the limits of the z-axis to include positive and negative values
        ax1.set_box_aspect((ncols, nrows, max(ncols, nrows)*0.8))                                                   # Adjust the 3D box aspect ratio for better visualization
        ax1.set_title('Matrix Values in 3D', fontweight='bold')                                                     # Set the title of the 3D subplot
        ax1.set_xlabel('GDLs')                                                                                      # Label the x-axis
        ax1.set_ylabel('GDLs')                                                                                      # Label the y-axis
        ax1.set_zlabel('Values')                                                                                    # Label the z-axis

        plt.tight_layout()                                                                                          # Adjust subplot spacing automatically
        plt.show()                                                                                                  # Display the final figure
        

#########################################################################################################################################
#########################################################################################################################################
################################################### Plot Global Displacement of RMF #####################################################
#########################################################################################################################################
#########################################################################################################################################


class PlotGlobalDislplacemet():
      def __init__(self, fs = 20, n_sub = 20,Lee_Nodos = any, delta_g_full = any, CoordNode = any, hv = 0.30, bc = 0.30, color = 'coolwarm'):
            self.fs = fs
            self.delta_g_full = delta_g_full
            self.CoordNode = CoordNode
            self.n_sub = n_sub
            self.hv = hv
            self.bc = bc
            self.color = color
            self.Lee_Nodos = Lee_Nodos 
      
      def PlotGD_RMF(self):

        fs = self.fs
        delta_g_full = self.delta_g_full
        CoordNode = self.CoordNode
        n_sub = self.n_sub
        bv = self.hv
        bc = self.bc
        color = self.color
        Lee_Nodos = self.Lee_Nodos

        delta_g_values = np.asarray(delta_g_full, dtype=float).ravel()

        # --- Hermite cubic shape functions for bending deformation ----------
        # v(s) = v_i*H1(s) + theta_i*L*H2(s) + v_j*H3(s) + theta_j*L*H4(s)
        def H1(s): return 1 - 3*s**2 + 2*s**3
        def H2(s): return s - 2*s**2 + s**3
        def H3(s): return 3*s**2 - 2*s**3
        def H4(s): return -s**2 + s**3

        s_vals = np.linspace(0, 1, n_sub)

        # Section thickness perpendicular to each element axis (visual width)
        # Elements 1,2 = beams (bv=0.30m). Elements 3,4,5 = columns (bc=0.40m).
        elem_thickness = [bv, bv, bc, bc, bc]

        # --- Colormap: displacement magnitude -> color (ETABS-like) --------
        cmap = plt.get_cmap(color)

        # --- Step 1: loop over all elements, compute deformed geometry ------
        # First pass to find global range of displacement magnitudes for
        # consistent color scaling across all elements.
        all_mags = []
        element_data = []

        for e, (n1, n2) in enumerate(Lee_Nodos):
                i = n1 - 1
                j = n2 - 1

                xi, yi = CoordNode[i, 0], CoordNode[i, 1]
                xj, yj = CoordNode[j, 0], CoordNode[j, 1]
                dx = xj - xi
                dy = yj - yi
                L_elem = np.sqrt(dx**2 + dy**2)
                cos_theta = dx / L_elem if L_elem > 0 else 1.0
                sin_theta = dy / L_elem if L_elem > 0 else 0.0

                ux_i, uy_i, rot_i = CoordNode[i, 2], CoordNode[i, 3], CoordNode[i, 4]
                ux_j, uy_j, rot_j = CoordNode[j, 2], CoordNode[j, 3], CoordNode[j, 4]

                # Transform global nodal displacements to local element axes:
                # u_local =  ux*cos + uy*sin   (axial)
                # v_local = -ux*sin + uy*cos   (transverse)
                # Rotation is invariant in 2D.
                ul_i =  ux_i * cos_theta + uy_i * sin_theta
                ul_j =  ux_j * cos_theta + uy_j * sin_theta
                vl_i = -ux_i * sin_theta + uy_i * cos_theta
                vl_j = -ux_j * sin_theta + uy_j * cos_theta

                u_interp_arr = np.zeros(n_sub)
                v_interp_arr = np.zeros(n_sub)
                x_local_def   = np.zeros(n_sub)
                y_local_def   = np.zeros(n_sub)

                for k, s in enumerate(s_vals):
                        # Hermite interpolation for transverse displacement (includes rotations)
                        v_interp_arr[k] = (vl_i * H1(s) + rot_i * L_elem * H2(s)
                                        + vl_j * H3(s) + rot_j * L_elem * H4(s))
                        # Linear interpolation for axial displacement
                        u_interp_arr[k] = ul_i * (1 - s) + ul_j * s
                        # Scaled deformed coordinates for visualization
                        x_local_def[k] = s * L_elem + u_interp_arr[k] * fs
                        y_local_def[k] = v_interp_arr[k] * fs

                # Deformed centerline in global coordinates
                x_def_curve = xi + x_local_def * cos_theta - y_local_def * sin_theta
                y_def_curve = yi + x_local_def * sin_theta + y_local_def * cos_theta

                # Normal vector (perpendicular to element axis)
                norm_x = -sin_theta
                norm_y =  cos_theta
                half_t = elem_thickness[e] / 2.0

                # Offset curves for the two sides of the member
                x_side1 = x_def_curve + norm_x * half_t
                y_side1 = y_def_curve + norm_y * half_t
                x_side2 = x_def_curve - norm_x * half_t
                y_side2 = y_def_curve - norm_y * half_t

                # Global displacement magnitude at each point along the element
                ux_global = u_interp_arr * cos_theta - v_interp_arr * sin_theta
                uy_global = u_interp_arr * sin_theta + v_interp_arr * cos_theta
                mags = np.sqrt(ux_global**2 + uy_global**2)
                all_mags.extend(mags.tolist())

                # Store for the drawing pass
                element_data.append({
                        "x_def": x_def_curve,
                        "y_def": y_def_curve,
                        "x_s1": x_side1, "y_s1": y_side1,
                        "x_s2": x_side2, "y_s2": y_side2,
                        "mags": mags,
                        "xi": xi, "yi": yi, "xj": xj, "yj": yj,
                        "cos": cos_theta, "sin": sin_theta,
                        "thick": elem_thickness[e],
                })

                print(f"elem_{e+1}, node i = {n1} to node j = {n2}")

        # --- Global range for colormap normalization -------------------------
        d_min = min(all_mags) if all_mags else 0
        d_max = max(all_mags) if all_mags else 1

        if d_max == d_min:
                d_max = d_min + 1e-12

        norm = plt.Normalize(d_min, d_max)
        print(f"Displacement range: {d_min*100:.4f} to {d_max*100:.4f} cm")

        # --- Compute bounding box from all deformed points ------------------
        # Both subplots will share the same axis limits so the deformation
        # scale is directly comparable between original and deformed shapes.
        # We collect all deformed centerline + offset curve points.
        all_x = []
        all_y = []
        for data in element_data:
                all_x.extend(data["x_def"])
                all_y.extend(data["y_def"])
                all_x.extend(data["x_s1"])
                all_y.extend(data["y_s1"])
                all_x.extend(data["x_s2"])
                all_y.extend(data["y_s2"])
        x_margin = 0.05 * (max(all_x) - min(all_x)) if all_x else 0.5
        y_margin = 0.05 * (max(all_y) - min(all_y)) if all_y else 0.5
        x_lim = [min(all_x) - x_margin, max(all_x) + x_margin]
        y_lim = [min(all_y) - y_margin, max(all_y) + y_margin]


        # --- Figure ----------------------------------------------------------

        fig, ax = plt.subplots(1, 2, figsize=(40, 10), constrained_layout=True)
        fig.suptitle("Frame Structure: Original and Deformed Shapes (with Rotations)",
                fontsize=16, fontweight="bold", color=(0,0,1))

        for a in ax:
                a.set_xlim(x_lim)
                a.set_ylim(y_lim)
                a.set_xlabel("X [m]")
                a.set_ylabel("Y [m]")

        # ==================================================================
        # LEFT SUBPLOT: original (undeflected) shape
        # ==================================================================
        for e, data in enumerate(element_data):
                xi, yi, xj, yj = data["xi"], data["yi"], data["xj"], data["yj"]
                cos_t, sin_t = data["cos"], data["sin"]
                half_t = data["thick"] / 2.0
                nx, ny = -sin_t, cos_t
                rect = np.array([
                        [xi + nx*half_t, yi + ny*half_t],
                        [xj + nx*half_t, yj + ny*half_t],
                        [xj - nx*half_t, yj - ny*half_t],
                        [xi - nx*half_t, yi - ny*half_t],
                ])
                ax[0].add_patch(plt.Polygon(rect, closed=True,
                        facecolor="#aaaaaa", edgecolor="#333333", lw=0.8, zorder=1))
                # Black centerline overlay (like the original line representation)
                ax[0].plot([xi, xj], [yi, yj], color=(0,0,0), lw=1.5, ls="-", zorder=6)
                mx, my = (xi + xj) / 2, (yi + yj) / 2
                ax[0].text(mx, my, f"E_{e+1}", fontsize=12,
                        ha="center", va="center", color=(1,1,1),
                        fontweight="bold", zorder=3)

        for n in range(len(CoordNode)):
                ax[0].text(CoordNode[n, 0], CoordNode[n, 1], f"N{n+1}",
                fontsize=12, ha="right", va="bottom", color=(1,0,0), zorder=4)
        ax[0].set_title("Original shape (with actual section width)")

        # ==================================================================
        # RIGHT SUBPLOT: deformed shape with displacement contour
        # ==================================================================
        for data in element_data:
                x_s1, y_s1 = data["x_s1"], data["y_s1"]
                x_s2, y_s2 = data["x_s2"], data["y_s2"]
                mags = data["mags"]

                # Gray undeflected reference (ghosted)
                xi, yi, xj, yj = data["xi"], data["yi"], data["xj"], data["yj"]
                cos_t, sin_t = data["cos"], data["sin"]
                half_t = data["thick"] / 2.0
                nx, ny = -sin_t, cos_t
                rect_undef = np.array([
                        [xi + nx*half_t, yi + ny*half_t],
                        [xj + nx*half_t, yj + ny*half_t],
                        [xj - nx*half_t, yj - ny*half_t],
                        [xi - nx*half_t, yi - ny*half_t],
                ])
                ax[1].add_patch(plt.Polygon(rect_undef, closed=True,
                facecolor="#dddddd", edgecolor="none", alpha=0.3, zorder=1))

                # Colored quads along the deformed element
                for k in range(n_sub - 1):
                        quad = np.array([
                        [x_s1[k],   y_s1[k]],
                        [x_s1[k+1], y_s1[k+1]],
                        [x_s2[k+1], y_s2[k+1]],
                        [x_s2[k],   y_s2[k]],
                        ])
                        color = cmap(norm(mags[k]))
                        ax[1].add_patch(plt.Polygon(quad, closed=True,
                        facecolor=color, edgecolor=color,
                        lw=0.5, alpha=0.95, zorder=2))

                # Black centerline overlay (like the original line representation)
                x_def, y_def = data["x_def"], data["y_def"]
                ax[1].plot(x_def, y_def, color=(0,0,0), lw=1.5, ls="-", zorder=6)

                # Deformed node markers (white-filled circles)
                ax[1].scatter([x_def[0], x_def[-1]], [y_def[0], y_def[-1]],
                        color=(0,0,0), s=50, facecolors=(1,1,1), zorder=5)

        ax[1].set_title(f"Deformed shape with displacement contour (scaled by fs = {fs})")

        # --- Colorbar ---------------------------------------------------------
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax[1], shrink=0.6, pad=0.02)
        cbar.set_label("Displacement magnitude [m]", fontsize=12)

        # --- Node displacement annotations on the deformed shape --------------
        for n in range(len(CoordNode)):
                x_node_def = CoordNode[n, 0] + CoordNode[n, 2] * fs
                y_node_def = CoordNode[n, 1] + CoordNode[n, 3] * fs
                ax[1].text(x_node_def, y_node_def,
                        f"d=({CoordNode[n, 2]*100:.2f}, {CoordNode[n, 3]*100:.2f}) cm\n"
                        f"$\\theta$={CoordNode[n, 4]*1e6:.2f} $\\times10^{{-6}}$ rad",
                        fontsize=12, ha="center", va="bottom", color=(0,0,1))

        plt.show()




#########################################################################################################################################
#########################################################################################################################################
########################################################## Plot Results Simple Beam #####################################################
#########################################################################################################################################
#########################################################################################################################################

class Manual_Flexural_Method():
    def __init__(self, am1 = any, am2 = any, am3 = any, am4 = any, am5 = any, X = any, Y = any
                 , ylabel = 'AM', L = any, title = 'Flexural'):
            self.am1 = am1
            self.am2 = am2
            self.am3 = am3
            self.am4 = am4
            self.am5 = am5
            self.X = X
            self.Y = Y
            self.ylabel = ylabel
            self.L = L
            self.title = title
            
    def Plot_AM_Manual_Flexural_Method(self):
            am1 = self.am1
            am2 = self.am2
            am3 = self.am3
            am4 = self.am4
            am5 = self.am5
            X = self.X
            Y = self.Y
            L = self.L
            ylabel = self.ylabel
            title = self.title
                            
            
            fig, ax = plt.subplots(6,1, figsize = (20,20))

            #----- all defformation------
            ax[0].plot(X, Y, color = (0,0,0), lw = 2, ls = '-', marker = 'o',
                    markersize = 10, markerfacecolor = (1,1,1))

            ax[0].plot(X, am1, color = (0,0,1), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, all defformations')
            ax[0].fill_between(X, am1, color = (0.5,0.5,1), alpha = 0.5)

            for i in np.arange(0, len(X),1):
                    ax[0].text(X[i],am1[i], f'{am1[i]:.2f}', fontsize = 10, fontweight = 'bold', ha='center', va='bottom')
                    
            ax[0].set_title(f'{title} Diagram, due all defformations', fontsize = 12, fontweight = 'bold')
            ax[0].set_xlabel('Distance [m]')
            ax[0].set_ylabel(ylabel)
            ax[0].set_xlim([-L*0.1,X[-1]+(L*0.1)])
            ax[0].legend()

            #----- Real loads ------
            ax[1].plot(X, Y, color = (0,0,0), lw = 2, ls = '-', marker = 'o',
                    markersize = 10, markerfacecolor = (1,1,1))

            ax[1].plot(X, am2, color = (1,0,0), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due Real Loads')
            ax[1].fill_between(X, am2, color = (1,0.5,0.5), alpha = 0.5)

            for i in np.arange(0, len(X),1):
                    ax[1].text(X[i],am2[i], f'{am2[i]:.2f}', fontsize = 10, fontweight = 'bold', ha='center', va='bottom')
                    
            ax[1].set_title(f'{title} Diagram, due Real Loads', fontsize = 12, fontweight = 'bold')
            ax[1].set_xlabel('Distance [m]')
            ax[1].set_ylabel(ylabel)
            ax[1].set_xlim([-L*0.1,X[-1]+(L*0.1)])
            ax[1].legend()

            #----- Climate Effects ------
            ax[2].plot(X, Y, color = (0,0,0), lw = 2, ls = '-', marker = 'o',
                    markersize = 10, markerfacecolor = (1,1,1))

            ax[2].plot(X, am3, color = (1,0,1), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due Climate Effects')
            ax[2].fill_between(X, am3, color = (1,0.5,1), alpha = 0.5)

            for i in np.arange(0, len(X),1):
                    ax[2].text(X[i],am3[i], f'{am3[i]:.2f}', fontsize = 10, fontweight = 'bold', ha='center', va='bottom')
                    
            ax[2].set_title(f'{title} Diagram, due Climate Effects', fontsize = 12, fontweight = 'bold')
            ax[2].set_xlabel('Distance [m]')
            ax[2].set_ylabel(ylabel)
            ax[2].set_xlim([-L*0.1,X[-1]+(L*0.1)])
            ax[2].legend()

            #----- due previus defformations ------
            ax[3].plot(X, Y, color = (0,0,0), lw = 2, ls = '-', marker = 'o',
                    markersize = 10, markerfacecolor = (1,1,1))

            ax[3].plot(X, am4, color = (0.5,0.5,1), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due previus defformations')
            ax[3].fill_between(X, am4, color = (0.25,0.25,0.5), alpha = 0.5)

            for i in np.arange(0, len(X),1):
                    ax[3].text(X[i],am4[i], f'{am4[i]:.2f}', fontsize = 10, fontweight = 'bold', ha='center', va='bottom')

            ax[3].set_title(f'{title} Diagram, due previus defformations', fontsize = 12, fontweight = 'bold')
            ax[3].set_xlabel('Distance [m]')
            ax[3].set_ylabel(ylabel)
            ax[3].set_xlim([-L*0.1,X[-1]+(L*0.1)])
            ax[3].legend()

            #----- due support Displacement ------
            ax[4].plot(X, Y, color = (0,0,0), lw = 2, ls = '-', marker = 'o',
                    markersize = 10, markerfacecolor = (1,1,1))

            ax[4].plot(X, am5, color = (0.3, 0.3, 0.3), lw = 1, ls = '--', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due support Displacement')
            ax[4].fill_between(X, am5, color = (0.3,0.3,0.3), alpha = 0.2)

            for i in np.arange(0, len(X),1):
                    ax[4].text(X[i],am5[i], f'{am5[i]:.2f}',fontsize = 10, fontweight = 'bold', ha='center', va='bottom')
                    
            ax[4].set_title(f'{title} Diagram, due support Displacement', fontsize = 12, fontweight = 'bold')
            ax[4].set_xlabel('Distance [m]')
            ax[4].set_ylabel(ylabel)
            ax[4].set_xlim([-L*0.1,X[-1]+(L*0.1)])
            ax[4].legend()

            #----- due support Displacement ------
            ax[5].plot(X, Y, color = (0,0,0), lw = 2, ls = '-', marker = 'o',
                    markersize = 10, markerfacecolor = (1,1,1))

            ax[5].plot(X, am1, color = (0,0,1), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, all defformations')
            ax[5].plot(X, am2, color = (1,0,0), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due Real Loads')
            ax[5].plot(X, am3, color = (1,0,1), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due Climate Effects')
            ax[5].plot(X, am4, color = (0.5,0.5,1), lw = 1, ls = '-', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due previus defformations')
            ax[5].plot(X, am5, color = (0.3, 0.3, 0.3), lw = 1, ls = '--', marker = '^', markersize = 4,
                    label = f'{title} Diagram, due support Displacement')


            ax[5].set_title(f'{title} Diagram, Comparison', fontsize = 12, fontweight = 'bold')
            ax[5].set_xlabel('Distance [m]')
            ax[5].set_ylabel(ylabel)
            ax[5].set_xlim([-L*0.1,X[-1]+(L*0.1)])
            ax[5].legend()

            plt.tight_layout() 
            plt.show()        

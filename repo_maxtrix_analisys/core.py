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
        # Tr = np.array([[1,0,0,0,0,0],
        #                [0,1,da,0,0,0],
        #                [0,0,1,0,0,0],
        #                [0,0,0,1,0,0],
        #                [0,0,0,0,1,-db],
        #                [0,0,0,0,0,1],
        #                ])
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
        thetha = np.deg2rad(self.thetha)                                                                            # Using orientation angle
        
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
#################################################### Managing Multiple 2D MF Elements ###################################################
#########################################################################################################################################
#########################################################################################################################################

class MF_L_elements2D:                                                                                              # Class for managing multiple 2D MF elements     
    def __init__(self):                                                                                             # Initialize the list of elements
        self.elements = []                                                                                          # List to store MF elements

    def add_element(self, element):                                                                                 # Method to add an element to the list                            
        self.elements.append(element)                                                                               # Append the element to the list                 

    def stacked_stiffness_matrices(self):                                                                           # Method to stack stiffness matrices of all elements         
        K_L_blocks = []                                                                                             # Initialize list to hold stiffness matrices                

        for elem in self.elements:                                                                                  # Loop through each element
            K_L_blocks.append(elem.stiffness_matrix_MF_L())                                                         # Append the stiffness matrix of the element

        return np.vstack(K_L_blocks)                                                                                # Return the stacked stiffness matrices as a single array
    
    def stacked_transformation_matrices(self):                                                                      # Method to stack transformation matrices of all elements         
        T_blocks = []                                                                                               # Initialize list to hold transformation matrices                

        for elem in self.elements:                                                                                  # Loop through each element
            T_blocks.append(elem.transformation_matrix_2D())                                                        # Append the transformation matrix of the element

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
        fig.suptitle("Representation of Stiffness Matrix Values",                                                   # Add a global title to the figure
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
########################################################## Plot Results Simple Beam ############################################################
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
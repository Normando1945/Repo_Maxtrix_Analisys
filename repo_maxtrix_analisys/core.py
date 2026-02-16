import numpy as np

#########################################################################################################################################
#########################################################################################################################################
############################ Stiffness Matrix for 2D MF Element with Rigid End Offsets and Shear Deformation ############################
#########################################################################################################################################
#########################################################################################################################################

class MF_K_T_L_Element2D:
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

    def stiffness_matrix_MF_L(self):
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
        beta = (6.0 * E * I) / (G * A * L**2) * f                                                                   # beta = (6EI)/(GA L^2) * f

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
        thetha = np.deg2rad(self.thetha)                                                                                        # Using orientation angle
        
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
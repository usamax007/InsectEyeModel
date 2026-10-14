# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:52:25 2024

@author: Usama
"""
import numpy as np

class CompoundEyeModel:
    
    threshIntensity = 0.5
    
    def __init__(self,species,angle_x,angle_y):
        # angle_x and angle_y are the total visual field angles in azimuthal and elevation respectively
        if species == 'Ms' or species == 'ms' or 'MS':
            print('Model initialized with Manduca sexta parameters')

            # Moth anatomical parameters
            self.radEye = 0.0041 / 2  # radius of compound eye ball (Theobald et al. 2010)
            self.d_phi_x = 0.96  # interommatidial angle (deg) (Theobald et al. 2010)
            self.d_phi_y = 0.96
            self.fpsVision = 125  # moth's vision frames per second
        else:
            raise Exception('unknown species name entered')
            
        # Compound eye size 
        self.n_phi_x = int(angle_x/self.d_phi_x)  # number of x facets (pixels)
        self.n_phi_y = int(angle_y/self.d_phi_y)  # number of y facets (pixels)
        
        self.visionMatrix_0 = [[]]
        self.visionMatrix_1 = [[]]
        
        self.phi_x_0, self.phi_y_0 = 0.0, 0.0
        # Grid of longitude (phi_x) and latitude (phi_y)
        self.phi_x_all = self._generateAngularGrid(self.phi_x_0,self.d_phi_x,self.n_phi_x)
        self.phi_y_all = self._generateAngularGrid(self.phi_y_0,self.d_phi_y,self.n_phi_y)
        
        # Grid of cartesian coordinates of gnomonic projection
        self.X, self.Y, self.Phi_x, self.Phi_y = [], [], [], []
        self._generateGnomonicProjection()
    
    def _generateAngularGrid(self,angle_0,d_angle,n_angles):
        array_temp = np.arange(d_angle / 2, d_angle*n_angles / 2, d_angle)
        return angle_0 + np.concatenate([-np.flip(array_temp),array_temp])
    
    def _generateGnomonicProjection(self):
        # x and y are in the units of compound eye radius
        for phi_y in self.phi_y_all:
            row_X, row_Y, row_Phi_x, row_Phi_y = [], [], [], []
            for phi_x in self.phi_x_all:
                x,y = self._forwardGnomonicProjection(self.phi_x_0,phi_x,self.phi_y_0,phi_y)
                row_X.append(x)
                row_Y.append(y)
                row_Phi_x.append(phi_x)
                row_Phi_y.append(phi_y)
            self.X.append(row_X)
            self.Y.append(row_Y)
            self.Phi_x.append(row_Phi_x)
            self.Phi_y.append(row_Phi_y)
    
    def _forwardGnomonicProjection(self,phi_0,phi,lam_0,lam): # give x and y
        theta = np.abs(np.arccos(np.sin(np.radians(phi)) * np.sin(np.radians(phi_0)) +
                                  np.cos(np.radians(phi)) * np.cos(np.radians(phi_0)) *
                                  np.cos(np.radians(lam_0 - lam))))
        y = self.radEye * np.cos(np.radians(phi_0)) * np.sin(np.radians(lam_0 - lam)) / np.cos(theta)
        x = -self.radEye * (np.cos(np.radians(phi)) * np.sin(np.radians(phi_0)) -
                        np.sin(np.radians(phi)) * np.cos(np.radians(phi_0)) *
                        np.cos(np.radians(lam_0 - lam))) / np.cos(theta)
        return x, y
    
    def giveVisualFieldMatrix(self,frameImage,mpx,mpy): # mpx is meters per pixel in x, mpy is meters per pixel in y
        n_py = np.size(frameImage,0)
        n_px = np.size(frameImage,1)
        cent_px = np.floor((n_px * 0.5))
        cent_py = np.floor((n_py * 0.5))
        scene_X, scene_Y = [], []
        for i_py in range(n_py):
            row_X, row_Y = [], []
            for i_px in range(n_px):         
                x = (i_px - cent_px) * mpx
                y = -(i_py - cent_py) * mpx
                row_X.append(x)
                row_Y.append(y)
            scene_X.append(row_X)
            scene_Y.append(row_Y)
        
        scene_px = -1*np.ones((n_py,n_px),int)
        scene_py = -1*np.ones((n_py,n_px),int)
        matrixVisualField = np.zeros(np.shape(self.X))
        count_matrixVisualField = np.zeros(np.shape(self.X))
        j_y_list = list(range(n_py))
        j_x_list = list(range(n_px))
        found_block = False
        for j_y in j_y_list:
            # print(j_y)
            for j_x in j_x_list:
                if not found_block:
                    i_y = 0
                    while i_y < len(self.X)-1:
                        i_x = 0
                        while i_x < len(self.X[0])-1:
                            if self._is_point_inside_quadrilateral_ray_casting((scene_X[j_y][j_x],scene_Y[j_y][j_x]), [(self.X[i_y][i_x],self.Y[i_y][i_x]),(self.X[i_y][i_x+1],self.Y[i_y][i_x+1]),(self.X[i_y+1][i_x+1],self.Y[i_y+1][i_x+1]),(self.X[i_y+1][i_x],self.Y[i_y+1][i_x])]):
                                print((i_y,i_x))
                                scene_px[j_y,j_x] = i_x
                                scene_py[j_y,j_x] = i_y
                                count_matrixVisualField[i_y,i_x] += 1
                                ind_x, ind_y = i_x, i_y
                                found_block = True
                                break
                            i_x += 1
                        if found_block:
                            break
                        i_y += 1
                else:
                    list_i_y = [ind_y, ind_y-1,ind_y+1]
                    list_i_x = [ind_x, ind_x-1,ind_x+1]
                    ii_y = 0
                    found_block_2 = False
                    while ii_y < 3:
                        i_y = list_i_y[ii_y]
                        ii_x = 0
                        while ii_x < 3:
                            i_x = list_i_x[ii_x]
                            if self._is_point_inside_quadrilateral_ray_casting((scene_X[j_y][j_x],scene_Y[j_y][j_x]), [(self.X[i_y][i_x],self.Y[i_y][i_x]),(self.X[i_y][i_x+1],self.Y[i_y][i_x+1]),(self.X[i_y+1][i_x+1],self.Y[i_y+1][i_x+1]),(self.X[i_y+1][i_x],self.Y[i_y+1][i_x])]):
                                #print((i_y,i_x))
                                scene_px[j_y,j_x] = i_x
                                scene_py[j_y,j_x] = i_y
                                count_matrixVisualField[i_y,i_x] += 1
                                ind_x, ind_y = i_x, i_y
                                found_block_2 = True
                                break
                            ii_x += 1
                        if found_block_2:
                            break
                        ii_y += 1
            j_x_list = list(reversed(j_x_list))    
        for j_py in range(n_py):
            for j_px in range(n_px):
                if scene_px[j_py,j_px] > -1 and scene_py[j_py,j_px] > -1: 
                    matrixVisualField[scene_py[j_py,j_px],scene_px[j_py,j_px]] += frameImage[j_py,j_px]/count_matrixVisualField[scene_py[j_py,j_px],scene_px[j_py,j_px]]
        self.visionMatrix_0 = self.visionMatrix_1
        self.visionMatrix_1 = matrixVisualField
        return(np.array(matrixVisualField).astype(np.uint8))
        
    def giveEventPolarityMatrix(self, makeBipolar):
        # Initialize variables
        matrixEventPolarity = np.zeros((self.n_phi_y,self.n_phi_x))
        curIntensity = np.log(np.array(self.visionMatrix_0+1.0, dtype=np.float64))
        nexIntensity = np.log(np.array(self.visionMatrix_1+1.0, dtype=np.float64))
    
        if makeBipolar:
            magNexIntensity = np.sqrt(nexIntensity ** 2)
            magCurIntensity = np.sqrt(curIntensity ** 2)
            difMagIntensity = magNexIntensity - magCurIntensity
            indsEvent = np.where(np.abs(difMagIntensity) > self.threshIntensity)
            matrixEventPolarity[indsEvent] = 255 * np.sign(difMagIntensity[indsEvent])
            matrixEventPolarity = matrixEventPolarity/2 + 128
            
        else:
            temp = nexIntensity - curIntensity
            logIntensity = np.sqrt(temp ** 2)
            indsEvent = np.where(logIntensity > self.threshIntensity)
            matrixEventPolarity[indsEvent] = 255
        
        return matrixEventPolarity.astype(np.uint8)
    
    def _is_point_inside_quadrilateral_ray_casting(self, point, quadrilateral):
        """
        Checks if a point is inside a quadrilateral using the ray casting algorithm.
    
        Args:
            point: A tuple (x, y) representing the point.
            quadrilateral: A list of four tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] representing the vertices of the quadrilateral in order.
    
        Returns:
            True if the point is inside the quadrilateral, False otherwise.
        """
        x, y = point
        intersections = 0
        for i in range(4):
            x1, y1 = quadrilateral[i]
            x2, y2 = quadrilateral[(i + 1) % 4]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                intersections += 1
        return intersections % 2 != 0
    
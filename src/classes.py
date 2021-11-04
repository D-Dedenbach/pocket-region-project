#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys, os
from copy import copy, deepcopy

cDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cDir)

import functions as fcs


class Patch():
    """
    Class representing a patch from a Fullerene surface manifold, a partial embedding built around a single pentagon node

    Attributes
    ----------
    
    """
    def __init__(self, pos=(0,0,0)):
        self.positions = fcs.generateDual(pos)
        self.global_index = np.zeros(self.positions.shape, dtype=int)
        
        self.triangles = np.array([[0,1,2],[0,2,3],[0,3,4], [0,4,5], [0,5,1]])
        self.dual_neighbours = fcs.generateDualNeighbours(self.triangles)
        
        self.rigid = np.zeros(self.positions.shape[0], dtype=bool)
        self.rigid[0:6] = True

    def __str__(self):
        return f"""
        Patch with center pentagon node of global index {self.global_index[0]}.
        """
    
    def addLayers(self, nLayers=1):
        """
        Adds layers to the patch.

        Parameters
        ----------
        nLayers : int, optional
            Specify number of layers added, by default 1
        """
        lastLayer = fcs.boundary(self.triangles)
        
        #Find layer index
        n = lastLayer.shape[0]
        l = int((n-5)/5 + 1)
        
        #Find lastLayer and currentLayer, lastCorner and currentCorner
        currentLayer = np.array([0])
        currentCorner = np.array([False])
        
        lastLayer = np.array([])
        lastCorner = np.array([])
        
        for i in range(1, l+1): 
            lastLayer = currentLayer
            currentLayer = fcs.nextLayer(self.dual_neighbours, currentLayer)
            
            #Corner arrays
            if i == 1:
                currentCorner = np.ones(5, dtype=bool)
                lastCorner = np.array([True])
                
            elif i > 1:
                lastCorner = currentCorner
                currentCorner = fcs.findCornerNodes(currentLayer, lastLayer, self.dual_neighbours)
                
                #Roll new layer such that first corner node has index 0
                currentLayer = np.roll(currentLayer, -fcs.firstTrue(currentCorner))
                currentCorner = np.roll(currentCorner, -fcs.firstTrue(currentCorner))
        
        #Now build the new layers
        for i in range(nLayers):
            
            #First: Building new layer, increase l
            l += 1
            
            #Compute Coordinates
            newCornerPositions = fcs.calc_newCornerPos(currentLayer[currentCorner], lastLayer[lastCorner], self.positions)
            newSectorPositions = fcs.calc_newSectorPos(newCornerPositions, l) 
            
            #Sort coordinates
            newNodes, nextCorner = fcs.orderNodeLayer(newCornerPositions, newSectorPositions)
            
            #Get next layer for triangles
            minIndex = self.positions.shape[0]
            nNewNodes = newNodes.shape[0]
            
            nextLayer = np.arange(minIndex, stop=minIndex+nNewNodes, dtype=int)
            
            newTriangles = fcs.TrianglesBetweenLayers(currentLayer, nextLayer)
            
            self.positions = np.vstack((self.positions, newNodes))
            self.triangles = np.vstack((self.triangles, newTriangles))
            self.dual_neighbours = fcs.generateDualNeighbours(self.triangles)
            
            lastLayer = currentLayer
            lastCorner = currentCorner
            
            currentLayer = nextLayer
            currentCorner = nextCorner                

class PocketRegion():
    '''
    Class representing a pocket region, an embedding of a Fullerene surface manifold region with high curvataure, combined from multiple patches.

    Attributes
    ----------
    positions:          3-dimensional positions of all dual nodes in the pocket region
    triangles:          all triangle faces  of the pocket region
    dual_neighbours:    all 6 neighbour nodes of each dual node
    global_index:       the global index of each dual node
    rigid:              bool value for each dual node; rigid nodes cannot be moved
    patches:            list of patches that the pocket region consists of
    layer_package:      
    '''
    def __init__(self, cPatch):
        self.positions = cPatch.positions
        self.triangles = cPatch.triangles
        self.dual_neighbours = cPatch.dual_neighbours
        self.global_index = cPatch.global_index
        
        self.rigid = np.zeros(self.positions.shape[0], dtype=bool)
        self.rigid[0:6] = True
        
        self.patches = [cPatch]

        self.layer_package = None
    
    def __str__(self):
        allPtg = self.global_index[self.global_index < 12]
        return f"""
        Pocket Region object holding {len(self.patches)} patches.
        The global indices of all contained pentagon nodes are {allPtg}.

        The current state of the layer package is:
        {self.layer_package}
        """
        
    def addPatches(self, patches):
        '''
        Add one or multiple patches to the pocket-internal list. This only appends the patch to the list, but does NOT take any geometric action.

        Parameters
        ----------
        patches: [patch object]
            List of patch objects or single patch object to be added.    
        '''
        if type(patches) == list:
            for patch in patches:
                self.patches.append(patch)
        else:
            self.patches.append(patches)
            
    def positionByFace(self, newPatch, orient):
        """
        Position newPatch such that the specfied face overlaps exactly with the specified face from the pocket region.
        First step in the process of merging.

        Parameters
        ----------
        newPatch : object of class cl.Patch()
            patch object to be positioned
        orient : [int]
            2x3 array. 0th row specifies node indices for pocket region, 1st row node indices for newPatch.
            Make sure to overlap the exact nodes, no permutations are allowed.
        """        
        #orient holds the indices of some overlapping nodes in aligning order. 3 nodes are needed to fix its position in 3d space
        ogNodes = self.positions[orient[0]].copy()
        newNodes = newPatch.positions[orient[1]].copy()
        
        #Reference point for the rotations is chosen as the 0-node of these arrays. translate both zero nodes to the same position
        vTr = ogNodes[0] - newNodes[0]
        fcs.translate(newPatch, vTr)
        
        #Define vectors 0->1 and 0->2 on each triangle. We need to paralellize 2 sets of vectors with 1 vector from each triangle
        vOg = np.array([ogNodes[1] - ogNodes[0], ogNodes[2] - ogNodes[0]])
        vNew = np.array([newNodes[1] - newNodes[0], newNodes[2] - newNodes[0]])
        
        #1st set of vectors is the normal vector of the triangles
        norm1 = np.cross(vOg[1], vOg[0])
        norm2 = np.cross(vNew[1], vNew[0])
        
        a = fcs.VectorAngle(norm1, norm2)
        
        #If angle is bigger than lower cutoff, rotate
        if a > 1e-3:
            rot = np.cross(norm1, norm2)
            fcs.rotate(newPatch, rVec=rot, refP=ogNodes[0], angle=-a)
            
        #update newPatch arrays, coordinates have changed
        newNodes = newPatch.positions[orient[1]].copy()
        vNew = np.array([newNodes[1] - newNodes[0], newNodes[2] - newNodes[0]])
        
        #Now norm1 and norm2 overlap. 2nd step is to overlap original vectors in the plane defined by the norm vectors
        a = fcs.VectorAngle(vOg[0], vNew[0])
        
        if a > 1e-3:
            rot = np.cross(vOg[0], vNew[0])
            if np.linalg.norm(rot) < 1e-3:
                rot = norm1
                
            fcs.rotate(newPatch, rVec=rot, refP = ogNodes[0], angle=-a)
            
    def fixGeometry(self, inputObj, method='edgelength'):
        """
        Collection of methods to fix the pocket region geometry when adding a new patch.
        Type of fixing is determined by method parameter.
        
        Uses Golden Section Search for optimization.

        Second step in the merging process, only used for sector merges.

        Parameters
        ----------
        inputObj : tuple
            Tuple of values depending on method:
                edgelength:     patchIndex, rotEdge, compareSelf, comparePatch
                symm_pointdist: patchIndex, rotEdge, compare0, compare1
                                Attention: symm_pointdist requires 2 patch indices and 2 rotEdge indices (as lists or np arrays)
                pointdist:      patchIndex, rotEdge, compareSelf, comparePatch
                
        method : string, optional
            Available methods at the moment:
                'edgelength'
                'symm_pointdist'
                'pointdist'
                
            The default is 'edgelength'.

        Raises
        ------
        SystemError
            When a non-existing method is specified.
            Existing methods: "edgelength", "pointdist" and "symm_pointdist".
        """
        if method == 'edgelength':
            #Unpack input
            patchIndex, rotEdge, compareSelf, comparePatch = inputObj
            
            #Get necessary data for rotation
            patch = self.patches[patchIndex]
            refP = patch.positions[rotEdge[0]].copy()
            rotVec = patch.positions[rotEdge[1]] - patch.positions[rotEdge[0]]
            
            def opt(angle):
                #Make rotation and extract the coordinate data we need for the edge length. 
                fcs.rotate(patch, rotVec, refP, angle)
                
                #Compute edge length:
                re = np.linalg.norm(self.positions[compareSelf] - patch.positions[comparePatch])
                
                #We want the distance between the points to be one.
                re = np.linalg.norm(re - 1)
                
                #Revert rotation, such that we are back at the original point
                fcs.rotate(patch, rotVec, refP, -angle)
                return re
            
            #Rotate back and forth between a and be to find optimal position
            a = -np.pi/2
            b = np.pi/2
            
            angleMin, angleMax, _ = fcs.goldenSection(opt, a, b, tol=1e-6)
            finalAngle = (angleMax + angleMin) / 2
            
            #Rotate to optimal position
            fcs.rotate(patch, rotVec, refP, finalAngle)
        
        elif method == 'symm_pointdist':
            
            patchIndex, rotEdge, compare0, compare1 = inputObj
            
            patch0 = self.patches[patchIndex[0]]
            patch1 = self.patches[patchIndex[1]]
            
            refP0 = patch0.positions[rotEdge[0,0]].copy()
            refP1 = patch1.positions[rotEdge[1,0]].copy()
            
            rot0 = patch0.positions[rotEdge[0,1]] - patch0.positions[rotEdge[0,0]]
            rot1 = patch1.positions[rotEdge[1,1]] - patch1.positions[rotEdge[1,0]]
            
            def opt(angle):
                #Make rotation and extract the coordinate data we need for the edge length. 
                fcs.rotate(patch0, rot0, refP0, angle)
                fcs.rotate(patch1, rot1, refP1, angle)
            
                #Compute edge length:
                re = np.linalg.norm(patch0.positions[compare0] - patch1.positions[compare1])
                
                #Revert rotation, such that we are back at the original point
                fcs.rotate(patch0, rot0, refP0, -angle)
                fcs.rotate(patch1, rot1, refP1, -angle)
                return re
             
            #Rotate back and forth between a and be to find optimal position
            a = -np.pi/2
            b = np.pi/2
            
            angleMin, angleMax, _ = fcs.goldenSection(opt, a, b, tol=1e-6)
            finalAngle = (angleMax + angleMin) / 2
            
            #Rotate to optimal position
            fcs.rotate(patch0, rot0, refP0, finalAngle)
            fcs.rotate(patch1, rot1, refP1, finalAngle)
            
            
        elif method == 'pointdist':
            #Unpack input
            patchIndices, rotEdge, comparePatch, compareRef = inputObj
            
            #Get necessary data for rotation
            patch = self.patches[patchIndices[0]]
            if patchIndices[1] == 0:
                refPatch = self
            else:
                refPatch = self.patches[patchIndices[1]]
                
            refP = patch.positions[rotEdge[0]].copy()
            rotVec = patch.positions[rotEdge[1]] - patch.positions[rotEdge[0]]
            
            def opt(angle):
                #Make rotation and extract the coordinate data we need for the edge length. 
                fcs.rotate(patch, rotVec, refP, angle)
                
                #Compute edge length:
                re = np.linalg.norm(refPatch.positions[compareRef] - patch.positions[comparePatch])

                #Revert rotation, such that we are back at the original point
                fcs.rotate(patch, rotVec, refP, -angle)
                return re
            
            #Rotate back and forth between a and be to find optimal position
            a = -np.pi/2
            b = np.pi/2
            
            angleMin, angleMax, _ = fcs.goldenSection(opt, a, b, tol=1e-6)
            finalAngle = (angleMax + angleMin) / 2
            
            #Rotate to optimal position
            fcs.rotate(patch, rotVec, refP, finalAngle)       
            
        else:
            raise SystemError("The method that was specified doesn't exist.")
        return
    
    def merge(self, newPatchID, overlap):
        """
        Merge the faces and nodes of 2 patches. 
        
        3rd step in the process after positionByFace and fixGeometry

        Parameters
        ----------
        newPatch : Dual class obj
            The new patch to add to the pocket region
        overlap : [int]
            integer array of size #overlappingFaces x 2 x 3. overlap[0] gives the faces as indexed in self.positions, overlap[1] as in newPatch.positions

        """
        
        newPatch = self.patches[newPatchID]
        #Step 3: merge positions and triangles
        
        #extract the necessary raw data that is to be modified
        newNodes = newPatch.positions.copy()
        newFaces = newPatch.triangles.copy()
        
        #create indexShift array that maps all indices of the new patch to their updated values in the pocket region index layer.
        indexShift = np.zeros(newPatch.positions.shape[0]).astype(int)
        nNew = 0
        
        for i in range(indexShift.shape[0]):
            graphNode = newPatch.global_index[i]
            
            if graphNode in self.global_index:
                pocketNode = fcs.reverseID(graphNode, self.global_index)
                indexShift[i] = pocketNode
            else:
                #Else, create new index
                indexShift[i] = np.max([np.max(self.triangles) + 1, np.max(indexShift) + 1])
                nNew += 1
        
        #Update the newFaces array with the index mapping
        for i in range(newFaces.shape[0]):
            for j in range(newFaces.shape[1]):
                idx = newFaces [i,j]
                newFaces[i,j] = indexShift[idx]
        
        #Extend positions array, as well as the self.rigid array, which notes nodes that are rigid in a patch; and the global_index array of the pocket
        positions = np.vstack((self.positions, np.zeros((nNew,3)))) 
        rigid = np.hstack((self.rigid, np.zeros(nNew, dtype=bool)))
        global_index = np.hstack((self.global_index, np.zeros(nNew, dtype=int)))
        
        #Compute new values
        for i in range(indexShift.shape[0]):
            positions[indexShift[i]] = newNodes[i]
            global_index[indexShift[i]] = newPatch.global_index[i]
        
        newRigid = fcs.transferRigidity(indexShift, newPatch.rigid)
        rigid[newRigid] = True
        
        
        #Update original arrays
        self.positions = positions
        self.rigid = rigid
        self.global_index = global_index
        
        
        #Update triangles
        delete = np.zeros(newFaces.shape[0]).astype(int)
        for i in range(overlap[:,0].shape[0]):
            idfier = np.any(newFaces==overlap[i,0,0], axis=1) * np.any(newFaces==overlap[i,0,1], axis=1) * \
                np.any(newFaces==overlap[i,0,2], axis=1)
            delete = delete|idfier
        
        newFaces = newFaces[delete == 0]
        allFaces = np.vstack((copy(self.triangles), newFaces))
        
        self.triangles = allFaces
        #self.dual_neighbours = fcs.generateDualNeighbours(self.triangles)          
                
class LayerPackage():
    '''
    Class representing a layer package, always keeping the current and the previous layer from both the global and local index system.

    Attributes
    ----------
    local_SAM:      local index dual_neighbours matrix
    global_SAM:     global index dual neighbours matrix
    current_local:  current local index layer
    current_global: corresponding current global index layer
    last_local:     previous local index layer
    last_global     corresponding previous global index layer

    Methods
    -------
    scale:          computes a new layer around the current one and shifts the current and last layer attributes           

    '''
    def __init__(self, local_SAM, global_SAM, local_first_layer, global_first_layer):
        self.local_SAM = local_SAM
        self.global_SAM = global_SAM                

        self._center = local_first_layer

        self._current_local = local_first_layer
        self._current_global = global_first_layer

        self._last_local = None
        self._last_global = None

    def __str__(self):
        return f"""
        Layer package center:
        {self._center}

        Current Layer:
        Local:
        {self._current_local}
        Global:
        {self._current_global}

        Last Layer:
        Local:
        {self._last_local}
        Global:
        {self._last_global}
        """
    
    @property
    def center(self):
        return self._center

    @property
    def current_local(self):
        return self._current_local

    @property
    def current_global(self):
        return self._current_global

    @property
    def last_global(self):
        return self._last_global

    @property
    def last_local(self):
        return self._last_local

    def scale(self, nLayers):
        for i in range(nLayers):
            self._last_local = self._current_local.copy()
            self._last_global = self._current_global.copy()

            self._current_local = fcs.nextLayer(self.local_SAM, self._current_local)
            self._current_global = fcs.nextLayer(self.global_SAM, self._current_global)
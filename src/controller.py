    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import classes as cl
import functions as fcs

class controller():
    '''
    The controller class assembles pocket regions and executes growing operations on them. 

    Attributes
    ----------
    __triangles: [int]
        global index surface manifold triangles array
    __dual_neighbours: [int]
        global index surface manifold dual neighbours array / sparse adjacency matrix
    name: str
        optional string to name the controller object. Default is an empty string
    pockets: [PocketRegion objects]
        List of PocketRegion objects in the controller object
    
    Methods
    -------
    groupAndGenerate:
        Groups all the pentagon nodes in __dual_neighbours, such that pocket regions can be defined.
        Then generates the patches and pocket region objects. Patches are finished after this step, whereas pocket regions are just started.

    merge:
        Merges all patches in a specified pocket together, such that the raw pocket is constructed.

    fillPocket:
        Creates what is called an intact layer structure in a pocket region. 
    '''
    def __init__(self, triangles, dual_neighbours, name=''):
        self.__triangles = triangles
        self.__dual_neighbours = dual_neighbours
        self.name = name
        self.pockets = []

    @property
    def triangles(self):
        return self.__triangles

    @property
    def dual_neighbours(self):
        return self.__dual_neighbours
    
    def groupAndGenerate(self):
        """
        Applies the grouping algorithm to self.dual_neighbours, generating the pentagon groups and the pocket and patch infrastructure.

        Returns
        -------
        None

        """
        #Distance matrix
        dist = fcs.ptgDist(self.__dual_neighbours)
        
        #Grouped pentagon nodes
        groups = fcs.groupPtgNodes(dist, self.__dual_neighbours)

        ####
        self.groups = groups
        self.dist = dist
        ####
        
        #Generate patches
        for group in groups:
            radii = fcs.planRadii(group, dist)
            patches = []

            for i in range(group.shape[0]):
                newPatch = cl.Patch()
                
                if radii[i] > 1:
                    newPatch.addLayers(radii[i] - 1)
                
                newPatch.globalIndex = fcs.idPatch(newPatch.dual_neighbours, self.__dual_neighbours, group[i])
                patches.append(newPatch)
            
            #Define pocket
            pocket = cl.PocketRegion(patches[0])
            self.pockets.append(pocket)
            
            pocket.addPatches(patches[1:])

    def merge(self, pocketID):
        '''
        Merges all patches in a given pocket region together. In the end, the raw pocket region with all crucial elements is finished, does not have an intact layer structure yet though.

        Parameters
        ----------
        pocketID: int
            Index of the pocket region in the controller.pockets list

        
        '''
        pocket = self.pockets[pocketID]
        patches = pocket.patches
        
        #Sanity check: Do groups exist?
        try:
            group = self.groups[pocketID]
            dist = self.dist
        except AttributeError:
            raise SystemError("Groups have not been defined yet. Please run .groupAndGenerate first.")

        #Reduce dist matrix to group nodes
        groupDist = dist[group, :][:, group]
            
        #Track-keeping and data holding structures
        positioned, active = np.zeros(len(group), dtype=bool), np.zeros(len(group), dtype=bool) 
        
        #Center patch is pocket, use as positional reference
        positioned[0] = True    
        
        while not np.sum(positioned) == positioned.shape[0]:
            
            #Find distance minima from all positioned nodes to all non-positioned nodes
            distMin = np.amin(groupDist[positioned,:][:,np.invert(positioned)])
            
            #get patches that are in minDist to any positioned patch
            activate = np.any(groupDist[:, positioned] == distMin, axis=1)
            
            #Make sure to activate only non-positioned patches
            for gIndex in range(activate.shape[0]):
                if activate[gIndex] and not positioned[gIndex]:
                    active[gIndex] = True

            #Position active patches          
            for graphIndex in group[active]:
                
                gIndex = np.where(group == graphIndex)[0][0]
                
                if positioned[gIndex]:
                    continue
                
                #Compute overlap to cPatch in pocket and patch indices and, if necessary, rotEdge in patch indices
                overlap, rotEdge = fcs.commonFaces(pocket, patches[gIndex], self.faces)
                
                #initial positioning of the new patch. Equates to step 1 of the 3-step patch merging process
                pocket.positionByFace(patches[gIndex], overlap[0])

                
                # =============================================================================#
                # Preliminary geometry fixing. Equates to step 2 of the 3-step patch merging process
                
                
                #2 cases: 2 overlap faces --> edge merge, no geometry fixing needed; 3 overlap faces --> flat merge, check 3rd party overlaps
                if overlap.shape[0] == 2:
                    positioned[gIndex] = True
                    #active[gIndex] = False
                
                elif overlap.shape[0] == 3:
                    positioned = fcs.geometryFixingParam(group, active, positioned, gIndex, patches, pocket, overlap, rotEdge, self.faces)
                    
                    #If no geometry fixing wanted for illustration etc., comment above and uncomment below
                    #positioned[gIndex] = True
                else:
                    #Other cases? Same as in the first case?
                    positioned[gIndex] = True
                    #active[gIndex] = False
                
        
                
            for graphIndex in group[active]:
                # =============================================================================#
                # Patch merging. Step 3 of the 3-step merging process    
                gIndex = np.where(group == graphIndex)[0][0]
                
                overlap, _ = fcs.commonFaces(pocket, patches[gIndex], self.faces)
                pocket.mergeFaces(gIndex, overlap)
                active[gIndex] = False
    
    def fillPocket(self, pocketID):
        '''
        WIP
        Attempts to create an intact layer structure in a pocket region by selecting a center and scaling the layers. 

        Parameters
        ----------
        pocketID: int
            Index of the pocket region object in the controller.pockets list.
        '''
        pocket = self.pockets[pocketID]
        
        #define array to save checked nodes. In certain situations that avoids a double check, which would result in error
        checked = np.zeros(pocket.nodes.shape[0], dtype=bool)
        
        
        centerNode = fcs.selectCenter(pocket.dual_neighbours, pocket.indexID, pocket.rigid)
        
        
        #Define first layer in graph and pocket indices
        graphLayer = pocket.indexID[centerNode]
        pocket.obtainLayerPackage(graphLayer, self.__dual_neighbours)
        nodeLayer = centerNode
        
        checked[nodeLayer] = True
        while True:
            pocket.scaleLayerPackage(self.__dual_neighbours)
            graphLayer = pocket.layerPackage[0]
            nodeLayer = fcs.nextLayer(pocket.dual_neighbours, nodeLayer)
            
            checked[nodeLayer] = True
            
            #break condition: no rigid nodes in layer
            if nodeLayer.shape[0] == 0 or np.sum(pocket.rigid[nodeLayer]) == 0:
                break
            
            #Check consistency: graph layer must have every index that nodeLayer has
            consistent = True
            for i in nodeLayer:
                if not pocket.indexID[i] in graphLayer:
                    consistent = False
            
            if consistent == False:
                raise SystemError('fillPocket: inconsistent layers: nodeLayer has indices that graphLayer doesn\'t')
            
            #Add missing nodes and fix determined nodes
            for x in graphLayer:
                pocketNode = fcs.reverseID(x, pocket.indexID)
                
                #If reverseID returns -1, the node x doesn't exist in pocket
                if pocketNode == -1:
                    #In that case, add the node and
                    pocketNode = self._addNodeToPocket(pocketID, x)
                
                #Check now, if the node is already rigid. If not, check for 3 rigid nb nodes, because then it is determined
                if not pocket.rigid[pocketNode]:
                    #find rigid neighbours
                    nb = pocket.dual_neighbours[pocketNode]
                    mask = pocket.rigid[nb]
                    
                    nb = nb[mask]
                    if nb.shape[0] >= 3:
                        nb = nb[:3]
                        
                        #3 rigid nb nodes -> get their coordinates and fix the node
                        p0 = pocket.nodes[nb[0]]
                        p1 = pocket.nodes[nb[1]]
                        p2 = pocket.nodes[nb[2]]
                        
                        pos = fcs.calc_vertex_position(p0, p1, p2, 1, 1, 1)
                        pocket.nodes[pocketNode] = pos
                        pocket.rigid[pocketNode] = True
                    
                    else:
                        continue
        return graphLayer
        
    def _addNodeToPocket(self, pocketID, graphNode):
        '''
        WIP
        Internal function that fully integrates a new node into a pocket region. 

        Parameters
        ----------
        pocketID: int
            controller.pockets index of the pocket region that the node should be added to

        graphNode: int
            Global index of the node that is added to the pocket region.
        '''
        pocket = self.pockets[pocketID]
        
        pocket.indexID = np.hstack((pocket.indexID, graphNode))
        pocket.rigid = np.hstack((pocket.rigid, False))
        
        ################## FACES ##################
        #find faces to add
        candidateFaces = []
        for face in self.faces:
            if graphNode in face:
                candidateFaces.append(face)
                
        candidateFaces = np.array(candidateFaces)
        mask = np.ones(candidateFaces.shape[0], dtype=bool)
        
        #change indexing to pocket index layer. If an index is not in pocket, don't keep the face
        for i in range(candidateFaces.shape[0]):
            for j in range(3):
                
                node = fcs.reverseID(candidateFaces[i, j], pocket.indexID)
                if node == -1:
                    mask[i] = False
                    break
                else:
                    candidateFaces[i,j] = node
        
        candidateFaces = candidateFaces[mask]
        pocket.faces = np.vstack((pocket.faces, candidateFaces))
        
        pocket.dual_neighbours = fcs.generateDualNeighbours(pocket.faces)
        
        ################## NODES ##################
        face = candidateFaces[0]
        
        #Added node has hightest index
        i = np.max(face)
        pos = np.where(face == i)[0][0]
        
        #bring i to face index 0
        face = np.roll(face, -pos)
        edge = face[1:]
        
        _, oppFace = fcs.searchForPlg(edge, pocket.faces)
        oppFace = pocket.faces[oppFace]
        
        mirrorNode = [x for x in oppFace if not x in edge][0]
        
        mirrorPos = pocket.nodes[mirrorNode]
        edgePos = pocket.nodes[edge]
        
        newNodePos = fcs.triangleFromEdge(edgePos, mirrorPos)
        
        pocket.nodes = np.vstack((pocket.nodes, newNodePos))
        
        #return node index
        return i
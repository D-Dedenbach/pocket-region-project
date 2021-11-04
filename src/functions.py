#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


# =============================================================================
# General Functions
# =============================================================================

#create initial pentagon
def startMesh(pos=(0,0,0), s=1):
    """
    Start a mesh with an equilateral pentagon

    Parameters
    ----------
    pos : [float], optional
        position of the pentagon center. The default is (0,0,0).
    s : float, optional
        pentagon edge length. The default is 1.

    Returns
    -------
    pent : [float]
        5 node coordinates in an array.

    """
    #calculate outer radius from edge length
    r = s / (2 * np.sin(np.pi / 5))
    pent = np.zeros((5,3))
    
    for i in range(0, 5):
        theta = i * 2*np.pi / 5
        
        pent[i, 0] = pos[0] + r * np.sin(theta) #circle around to get x and y coordinates
        pent[i, 1] = pos [1] + r * np.cos(theta)
        pent[i, 2] = pos[2] + 0 #z is zero, initial pentagon is in flat plane
    
    return pent

def rotateVector(v, rot, angle, direc='CCW'):
    """
    rotate a vector around a rotator vector.

    Parameters
    ----------
    v : [float]
        Vector to rotate.
    rot : [float]
        Rotator vector. Will get normalized, only direction important.
    angle : float
        Angle to rotate the vector with.
    direc : str, optional
        Rotation direction. The default is 'CCW'. Second option is 'CW'.

    Returns
    -------
    rotated_vec : [float]
    
    """
    from scipy.spatial.transform import Rotation as R
    
    if direc == 'CW':
        angle = - angle
    
    rot = rot / np.linalg.norm(rot)
    rotation_vector = angle * rot

    rotation = R.from_rotvec(rotation_vector)

    rotated_vec = rotation.apply(v)
    
    return rotated_vec

def plgCenter(verts):
    """
    Calculate averaged plg center

    Parameters
    ----------
    verts : [float] (Nxd)
        node positions.

    Returns
    -------
    center: [float] (1xd)
        Center position.

    """
    return np.sum(verts, axis=0) / verts.shape[0]

def goldenSection(f, a, b, tol):
    """
    Golden section optimization.

    Parameters
    ----------
    f : function
        Function to optimize. One required input parameter.
    a : float
        Minimal input parameter value.
    b : float
        Maximal input parameter value.
    tol : float
        Combined relative and absolute tolerance.

    Returns
    -------
    a : float
        Lower boundary of end result for input parameter.
    b : float
        Upper boundary of end result for input parameter.
    counter : int
        Number of iterations run.

    """
    tau = (np.sqrt(5) - 1) / 2
    x_1 = a + (1-tau)*(b - a)
    f_1 = f(x_1)
    
    x_2 = a + tau *(b - a)
    f_2 = f(x_2)
    
    counter = 0
    while((b - a) > tol):
        counter += 1
        
        if f_1 > f_2:
            a = x_1
            x_1 = x_2
            f_1 = f_2
            x_2 = a + tau*(b - a)
            f_2 = f(x_2)
        
        else:
            b = x_2
            x_2 = x_1
            f_2 = f_1
            x_1 = a + (1 - tau)*(b - a)
            f_1 = f(x_1)
        
    return a, b, counter

def VectorAngle(vector_1, vector_2):
    """
    Calculate angle between vectors

    Parameters
    ----------
    vector_1 : [float]
        First vector.
    vector_2 : [float]
        Second vector.

    Returns
    -------
    angle : float

    """
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1) 
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    
    if np.linalg.norm(unit_vector_1 + unit_vector_2) < 1e-3:
        return np.pi
    
    if np.linalg.norm(unit_vector_1 - unit_vector_2) < 1e-3:
        return 0
    
    dot_product = np.dot(unit_vector_1, unit_vector_2) 
    
    angle = np.arccos(dot_product)
    
    return angle

#Given a half-edge, find the faces that it is part of
def searchForPlg(edge, faces):
    """
    Given a half-edge, find the faces that it is part of.

    Parameters
    ----------
    edge : [int]
        The half-edge (=directed edge)
    faces : [int]; (3,)
        all triangles to search in.
        

    Returns
    -------
    plg : int
        index of the plg the half edge is part of
    oppPlg : int
        index of the plg the opposite half edge is part of
    """
    plg = -10
    oppPlg = -10
    
    for i in range(faces.shape[0]):
        face = faces[i]
        
        #Remove -1's from pentagons
        if np.where(face == -1)[0].shape[0] != 0:
            face = face[face >= 0]
        
        for j in range(face.shape[0]):
            
            testEdge = [face[j], face[(j+1)%face.shape[0]]]
            
            if testEdge[0] == edge[0] and testEdge[1] == edge[1]:
                plg = i
            elif testEdge[0] == edge[1] and testEdge[1] == edge[0]:
                oppPlg = i
    return plg, oppPlg

#Check if an edge exists. When it is a boundary edge, direction is important, so return that info
def edgeExist(edge, faces):
    '''
    Check, if an edge exists. If it is a boundary edge, direction in important, so return that information.

    Parameters
    ----------
    edge: [int]
        pair of integers that build the edge
    faces: [int]
        array of dual faces
    
    Returns
    -------
    isInFaces: int
        integer indicating how many times the undirected edge is in the manifold
    swap: bool  
        if isInFaces is 1, this indicates whether the edge is in the same order as the input (False) or in the opposite order (True).  
    '''
    inFaces= np.any(faces == edge[0], axis=1) * np.any(faces == edge[1], axis=1)
    if np.sum(inFaces) == 1:
        face = faces[inFaces][0]
        identifier = np.where(face == edge[0])[0][0]
        if not face[(identifier+1)%face.shape[0]] == edge[1]:
            swap = True
        else:
            swap = False
    else:
        swap = False
    return  np.sum(inFaces), swap


def shift(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]

#similar to np.unique for 1d arrays, just keeps the ordering
def unique1D(a):
    '''
    Returns unique entries of the input array in order.
    Similar to np.unique, just keeps the ordering.
    '''
    (un, idfier) = np.unique(a, return_index=True)
    
    
    ordered = False
    while not ordered:
        ordered = True
        i = 0
        for i in range(idfier.shape[0]-1):
            if idfier[i] > idfier[i+1]:
                
                idfier[[i, i+1]] = idfier[[i+1, i]]
                un[[i, i+1]] = un[[i+1, i]]
                
                ordered = False
    
    return un   

def reverseID(graphNode, global_index_array):
    '''
    This finds the local index to a given global index.

    Parameters
    ----------
    graphNode: int
        The global node index that needs a local equivalent
    global_index_array: [int]
        The global_index array from a pocket region or patch, that the local node index is required for

    Returns
    -------
    node: int

    '''
    
    potNodes = np.where(global_index_array == graphNode)[0]
    
    if len(potNodes) == 0:
        node = -1
    else:
        node = potNodes[0]
    
    return node

# =============================================================================
# Testing- and Debugging-specific
# =============================================================================
def getPocketPentagons(pocket):
    '''
    Returns the global indices of all pentagon nodes in the pocket.

    Parameters
    ----------
    pocket: PocketRegion object 
    '''
    pentagons = []
    for i in range(pocket.indexID.shape[0]):
        if pocket.global_index[i] < 12:
            pentagons.append(i)
    
    return pentagons
    
def emergencyPlot(nodes, faces, plotLabs=False):
    '''
    Quickly make a vedo plot of a set of coordinates and triangles, for visual checkup.

    Parameters
    ----------
    nodes:  [float]
        node positions array
    faces: [int]
        faces array
    plotLabs: bool
        plot labels on the nodes, default is False
    '''
    import vedo
    
    vp = vedo.Plotter()
    mesh = vedo.Mesh([nodes, faces])
    mesh.lineColor('red').lineWidth(2).alpha(0.7)
    labs = mesh.labels('id')
    
    if plotLabs == False:
        vp.show(mesh)
    
    else:
        vp.show(mesh, labs)
    
    vedo.plotter.closePlotter()

def computeDeficitAngle(node, pocket):
    '''
    Returns the deficit angle divided by the theoretical deficit angle. Only for ptg nodes at this time.

    Parameters
    ----------
    node : int
        Local index of the node, around which the deficit angle is to be calculated.
    pocket : PocketRegion  object
        Pocket region with all the data.

    Returns
    -------
    deficitAngle: float
        Deficit angle divided by full angle.

    '''
    nb = pocket.dual_neighbours[node]
    nb = nb[nb>=0]
    
    angle = 0
    for i in range(nb.shape[0]):
        v1 = pocket.nodes[nb[i]] - pocket.nodes[node]
        v2 = pocket.nodes[nb[(i+1)%nb.shape[0]]] - pocket.nodes[node]
        
        angle += VectorAngle(v1, v2)
    
    deficitAngle = 2 * np.pi - angle
    
    return deficitAngle * 6 / (2 * np.pi)

def checkEdgeLengths(pocket):
    dev = 0
    dev_edges = []
    lengths = []
    
    edges = []
    for face in pocket.faces:
        for i in range(3):
            edge = [face[i], face[(i+1)%3]]
            if not edge in edges or [edge[1], edge[0]] in edges:
                edges.append(edge)
    
    for edge in edges:
        l = np.linalg.norm(pocket.nodes[edge[0]] - pocket.nodes[edge[1]])
        
        if not np.isclose(l, 1):
            dev += 1
            dev_edges.append(edge)
            lengths.append(l)
            
    print(dev, 'edges deviated from 1 substantially (atol=1e-8, rtol=1e-5, summed up tol)')
    if not len(dev_edges) == 0:
        print(dev_edges)
        print(lengths)
# =============================================================================
# Dual mesh construction
# =============================================================================
def generateDual(pos=(0,0,0)):
    """
    generate a dual pentagon that is rotated by 2 pi compared to the startMesh pentagon.

    Parameters
    ----------
    pos : [float], optional
        ptg center coordinates. The default is (0,0,0).

    Returns
    -------
    dualGrid : [float]
        5 node coordinates in an array.

    """
    pos = np.array(pos)
    zPos = pos +  np.array((0,0,np.sqrt(1 - 1/(4*np.sin(np.pi/5)**2))))
    dualGrid = startMesh(pos=zPos)


    #Mirror the dual pentagon along x- and y-axis so it works out with the primal mesh.
    dualGrid[:,1] = - dualGrid[:, 1]
    dualGrid[:,0] = - dualGrid[:,0]
    #roll the array so we still begin in the top left
    dualGrid = np.roll(dualGrid,2,axis=0)
    
    dualGrid = np.vstack((pos, dualGrid))
    
    return dualGrid

def dualMesh(pVerts, pFaces):
    """
    Constructs a dual mesh from a cubic mesh

    Parameters
    ----------
    pVerts : (,3)
        Cubic node coordinates.
    pFaces : (,6)
        Cubic faces.

    Raises
    ------
    SystemExit
        When it doesn't find the next node in a polygon.

    Returns
    -------
    dVerts : (,3)
        Dual node coordinates.
    dFaces : (,3)
        Dual faces.

    """
    dVerts = np.zeros((pFaces.shape[0], 3))
    dFaces = []
    
    #dual faces. Begin by circling through all edges in the primal mesh
    for i in range(pFaces.shape[0]):
        pFace = pFaces[i]
        
        
        if np.where(pFace == -1)[0].shape[0] != 0:
            pFace = pFace[pFace >= 0]
        
        dVerts[i, :] = plgCenter(pVerts[pFace])
        
        for j in range(pFace.shape[0]):
            
            #get vertex indices that define the edge
            edge = [pFace[j], pFace[(j+1)%pFace.shape[0]]]
            #first node of the face is simply the polygon we are in
            face = [i]
            
            #now search for the adjacent polygon with the opposite half-edge
            node, _ = searchForPlg(edge[::-1], pFaces)
            face.append(node)
            
            if node < 0:
                continue
            
            #go to the neighboring half-edge in that polygon, in clockwise direction
            index = np.where(pFaces[node, :] == edge[1])[0]
            
            if index.shape == (0,):
                raise SystemExit('index is []')
           
            
            edge = [pFaces[node][(index[0]-1)%pFaces[node].shape[0]], pFaces[node][index[0]]]    
            
            
            #find the opposite half-edge of that edge, which is part of the third polygon
            node, _ = searchForPlg(edge[::-1], pFaces)
            
            if node < 0:
                continue
            face.append(node)
            
            if not (face in dFaces or shift(face, 1) in dFaces or shift(face, 2) in dFaces):
                dFaces.append(face)
            
    dFaces = np.array(dFaces)
    return dVerts, dFaces

#Construct a triangle (standard angle: equilateral, otherwise isosceles) from an edge and a reference point for direction.
def triangleFromEdge(edgeVerts, direcPt, angle=np.pi/3):
    """
    Generates a triangle based on a mirroring along the given edge. Useful when mirroring a face to construct positions for new nodes.

    Parameters
    ----------
    edgeVerts : (2,3)
        Matrix of positions for 2 nodes that define the edge.
    direcPt : (1,3)
        reference point coordinates.
    angle : float, optional
        Optional give the rotation angle (angle between edge[0]/edge[1] and edge[0]/newPt. The default is np.pi/3.

    Returns
    -------
    newNode : (1,3)
        New node positions.

    """
    edge = edgeVerts[1] - edgeVerts[0]
    direc = direcPt - edgeVerts[0]
    
    rot = np.cross(edge, direc)
    
    newNode = edgeVerts[0] + rotateVector(edge, rot, -angle) 
    
    return newNode

#Compute the edges that belong to the boundary of a patch.
def boundary(faces):
    '''
    find the boundary of a patch or pocket region

    Parameters
    ----------
    faces: [int]
        patch.triangles or PocketRegion.triangles array
    
    Returns
    -------
    orderedBdr: [int]
        CCW ordered boundary nodes
    '''
    minNode = np.amin(faces)
    maxNode = np.max(faces)
    
    edges=[]
    for i in range(minNode, maxNode+1):
        for j in range(i, maxNode+1):
            
            if i == j:
                continue
            
            edge=[i,j]
            check, swap = edgeExist(edge, faces)
            
            if swap:
                edge=np.array([j,i])
                
            if check == 1:
                
                edges.append(edge)
    
    edges = np.array(edges)
    orderedBdr = []
    #start with first edge
    nextEdge = edges[0]
    while True:
        orderedBdr.append(nextEdge)
        identifier = edges[:,0] == nextEdge[1]
        nextEdge = edges[identifier]
        
        if len(nextEdge) ==0:
            raise SystemExit('Error in boundary array. Couldnt find next edge in closed boundary.')
        else:
            nextEdge = nextEdge[0]
        
        if np.all(nextEdge == edges[0]):
            break
    
    return np.array(orderedBdr)     

def linkClosed(node, faces):
    idfier = np.any(faces == node, axis=1)
    relevantFaces = faces[idfier]
    
    allNodes = np.unique(relevantFaces)
    re = True
    for i in allNodes:
        if i == node:
            continue
        else:
            if not np.sum(relevantFaces == i) == 2:
                re = False
                break
    
    return re

#Find the link of a vertex in dual rep, return it in ccw order.
def neighbours(vertex, faces):
    #choose all faces that vertex is a part of
    nb = faces[np.sum(faces == vertex, axis=1) > 0]
    
    #roll the vertex to pos 0 in the row
    for k in range(nb.shape[0]):
        n = np.where(nb[k] == vertex)[0][0]
        nb[k] = np.roll(nb[k], -n)
    
    #take the edges that do not contain 0
    linkEdges = []
    for i in range(nb.shape[0]):
        linkEdges.append([nb[i,1], nb[i,2]])
    
    linkEdges = np.array(linkEdges)
    
    #Check if link is closed
    closed = linkClosed(vertex, nb)
    """
    if linkEdges.shape[0] >= 5:
        closed = True
    else:
        closed = False
    """
    #If link has to be ordered; method differs depending on closed.
    if closed:
        orderedLink = []
        
        #If link is closed we start with first entry
        nextEdge = linkEdges[0]

        while True:
            orderedLink.append(nextEdge)
            identifier = linkEdges[:,0] == nextEdge[1]
            nextEdge = linkEdges[identifier]
            
            if len(nextEdge) ==0:
                raise SystemExit('Error in link array. couldnt find next edge in closed link.')
            else:
                nextEdge = nextEdge[0]
            
            if np.all(nextEdge == linkEdges[0]):
                break
        
        linkEdges = np.array(orderedLink)
    
    #If link is not closed, order:
    else:
        orderedLink = []
        
        #If link is not closed, check starting edge
        for i in linkEdges[:,0]:
                if not np.any(linkEdges[:,1] == i):
                    nextEdge = linkEdges[linkEdges[:,0] == i][0]
                    break
        
        while True:
            orderedLink.append(nextEdge)
            identifier = linkEdges[:,0] == nextEdge[1]
            nextEdge = linkEdges[identifier]
            
            #Here we use the fact that at some point there will be no next edge to end the sorting
            if len(nextEdge) ==0:
                break
            else:
                nextEdge = nextEdge[0]
        
        linkEdges = np.array(orderedLink)
        
    linkEdges = unique1D(linkEdges)
    
    return linkEdges

def generateDualNeighbours(faces):
    nMax = np.amax(faces) + 1
    
    dual_neighbours = np.ones((nMax, 6), dtype=int) * -1
    
    for i in range(nMax):
        nb = neighbours(i, faces)
        
        for j in range(nb.shape[0]):
            dual_neighbours[i, j] = nb[j]
    
    return dual_neighbours

# =============================================================================
# Cubic mesh construction
# =============================================================================

def constructCubicFaces(dualFaces):
    nodes = np.max(dualFaces) + 1
    cubicFaces = []
    
    for i in range(nodes):
        #If a node is in less than 5 faces, it is on the border and we don't want a cubic face around it
        if np.sum(np.any(dualFaces == i, axis=1)) < 5 or not linkClosed(i, dualFaces):
            continue
        
        else:
            idfier = np.any(dualFaces == i, axis=1)
            relevantFaces = dualFaces[idfier]
            relevantIndices = np.nonzero(idfier)[0]
            
            #bring node index to first pos
            for j in range(relevantFaces.shape[0]):
                
                roll = np.where(relevantFaces[j] == i)[0][0]
                relevantFaces[j] = np.roll(relevantFaces[j], -roll)
                
            #construct cubic face
            cubicFace = [relevantIndices[0]]
            faceIndex=0
            while True:
                definingNode = relevantFaces[faceIndex, 2]   
                
                idfier = relevantFaces[:,1] == definingNode
                faceIndex = np.nonzero(idfier)[0][0]
                
                if relevantIndices[faceIndex] == cubicFace[0]:
                    cubicFaces.append(cubicFace)
                    break
                else:
                    cubicFace.append(relevantIndices[faceIndex])
    cubicFaces = np.array(cubicFaces, dtype=object)
    return cubicFaces

# =============================================================================
# Patch positioning
# =============================================================================
def translate(obj, tVec):
    obj.nodes[:,0] = obj.nodes[:, 0] + tVec[0]
    obj.nodes[:,1] = obj.nodes[:, 1] + tVec[1]
    obj.nodes[:,2] = obj.nodes[:, 2] + tVec[2]
    

def rotate(obj, rVec, refP, angle):
    """
    Rotates the object around a vector with a specified position

    Parameters
    ----------
    rVec : [float]
        Vector that is rotated around.
    refP : [float]
        point in space that the vector is positioned on.
    angle : float
        Angle of rotation around the vector. Right hand rule applies.

    Returns
    -------
    None. Patch coordinates are updated.

    """
    #first, translate grid such that refP is origin
    translate(obj, -refP)
    
    #rotate around origin
    for i in range(obj.nodes.shape[0]):
        obj.nodes[i] = rotateVector(obj.nodes[i], rVec, angle)
    
    #translate back
    translate(obj, refP)
        
# =============================================================================
# Controller and Growth 
# =============================================================================

def nextLayer(dualNeighbors, curLayer):
    """
    Gives back the nodes of the next Layer of faces, defined by a CCW set of nodes. 

    Parameters
    ----------
    dualNeighbors : (,6)
        Dual neighbors generated from graph of the desired molecule.
    curBdr : (,1)
        Set of nodes that defines the current Layer by a CCW ordered boundary of points, array of indices.

    Returns
    -------
    nodes : (,1)
        New boundary nodes.

    """
    #make sure that curLayer has the correct type
    if isinstance(curLayer, int) or isinstance(curLayer, np.int64):
        curLayer = np.array([curLayer])
    elif isinstance(curLayer, list):
        curLayer = np.array(curLayer)
        
    #Get all node connections to the outside of every boundary node
    potNodes = []
    
    if curLayer.shape[0] == 1:
        nodes = dualNeighbors[curLayer[0]]
        nodes = nodes[nodes != -1]
        return nodes
    
    else:
        for i in range(curLayer.shape[0]):
            c = dualNeighbors[curLayer[i]]
            
            l, r = curLayer[(i-1)%curLayer.shape[0]], curLayer[(i+1)%curLayer.shape[0]]
            
            #If the point is part of a dead end ("pockets" of the circle closing), it won't have l and r in its neighbors
            if not l in c or not r in c:
                continue
            
            #Circle through neighbors to extract everything between r and l
            j = 0
            append = False
            
            while True:
                if append and c[j%6] == r:
                    break
                elif c[j%6] == l:
                    append = True
                elif append:
                    potNodes.append(c[j%6])
                
                j += 1
        
        #Select unique entries
        nodes = []
        for x in potNodes:
            #x in nodes is for uniqueness, x in layer is for sharp corners, such that we do not take indices from the curLayer for the newLayer.
            if not x in nodes and not x in curLayer:
                nodes.append(x)
        
        nodes = np.array(nodes)
        nodes = nodes[nodes >= 0]
        
        return nodes

def ptgDist(dualNeighbors):
    """
    Generates a matrix that holds all distances between ptg nodes

    Parameters
    ----------
    dualNeighbors : (,6)
        Array of dual neighbors.

    Returns
    -------
    dist : (12,12)
        Matrix of distances between the 12 pentagon nodes.

    """
    
    dist = np.zeros((12,12), dtype=int)
    
    #Circle through all pentagons
    for i in range(12):
        #Count for the number of pentagons that appeared already, layercount for counting layers
        count = 1
        layercount = 0
        
        #first layer is the ptg node
        layer = [i]
        while count<12:
            #Go to next layer
            layercount += 1
            layer = nextLayer(dualNeighbors, layer)
            
            #If ptg is in that layer, note that in the dist array
            for j in layer:
                if j<12:
                    count += 1
                    
                    dist[i, j] = layercount
    return dist

def rigidDist(dualNeighbours, rigid):
    """
    Generates a distance matrix of all rigid nodes in a pocket region

    Parameters
    ----------
    dualNeighbours : (,6)
        Array of dual neighbours.
    rigid : (1,)
        Bool array that holds, which nodes are rigid and which aren't.

    Returns
    -------
    dist : (nRigid, nRigid)
        Distance matrix of rigid nodes.

    """
    nRigid = np.sum(rigid)
    dist = np.zeros((nRigid, nRigid), dtype=int)
    
    #Enumerate to get the index values of the rigid nodes out
    indices = [i for i, x in enumerate(rigid) if x]
    
    for distIndex, i in enumerate(indices):
        count = 1
        layercount = 0
        
        layer = [i]
        
        while count < nRigid:
            layercount += 1
            layer = nextLayer(dualNeighbours, layer)
            
            for j in layer:
                if j in indices:
                    count += 1
                    
                    dist[distIndex, np.where(indices == j)[0][0]] = layercount
    
    return dist
            
############################## groupPtgNodes ###############################
def nodeCluster(cNode, dual_neighbours, distMax):
    tocheck = [cNode]
    cluster = [cNode]
    index = 0
    
    while not len(tocheck) == index:
        c = tocheck[index]
        
        nb = nextLayer(dual_neighbours, c)
        
        if distMax > 1:
            layer = nb
            for k in range(1, distMax):
                layer = nextLayer(dual_neighbours, layer)
                nb = np.hstack((nb, layer))
        
        for node in nb:
            if node >= 0 and node < 12 and not node in tocheck and not len(cluster) > 5:
                tocheck.append(node)
                cluster.append(node)
        
        index += 1
    cluster = np.array(cluster)
    
    return cluster
                
        
        
    return
def avgDist(idx, ref, dist):
    if isinstance(ref, list):
        ref = np.array(ref)
    
    allDists = dist[idx, ref]
    avg = np.sum(allDists) / ref.shape[0]
    
    return avg

def avgDistMin(dist):
    idfier = dist > 0
    dist = dist[idfier].reshape(11,12)
    
    avgMin = np.sum(np.amin(dist, axis=1)) / 12
    
    return avgMin

def minCount(dist, node):
    distMin = np.min(dist[dist > 0])
    
    count = np.sum(dist[node] == distMin)
    return count
 
def minGroupDist(group1, group2, dist):
    """
    Calculates the minimum distance between 2 groups

    Parameters
    ----------
    group1 : (), int
        group or node one.
    group2 : (), int
        group or node two.
    dist : (12,12)
        pentagon distance matrix.

    Returns
    -------
    minDist : int
        minimum distance between groups.

    """
    if isinstance(group1, int) or isinstance(group1, np.int64):
        group1 = np.array(group1)
    if isinstance(group2, int) or isinstance(group2, np.int64):
        group2 = np.array(group2)
    
    minDist = np.inf
    
    for i in range(group1.shape[0]):
        for j in range(group2.shape[0]):
            if dist[group1[i], group2[j]] < minDist:
                minDist = dist[group1[i], group2[j]]
    return minDist


def sortGroups(groups, dist):
    for group in groups:
        if group.shape[0] == 1:
            continue
        
        #Define and fill an average distance array inside the group members
        avg = np.zeros(group.shape[0])
        for i in range(group.shape[0]):
            avg[i] = avgDist(group[i], group[group != group[i]], dist)
        
        #Simple ordering algo
        ordered = False
        while not ordered:
            ordered = True
            for j in range(avg.shape[0] - 1):
                if avg[j] > avg[j+1]:
                    ordered = False
                    
                    avg[j+1], avg[j] = avg[j], avg[j+1]
                    group[j+1], group[j] = group[j], group[j+1]  
    
    return groups


def undecidedIndices(groups, dist, active):
    for i in range(12):
        if active[i] == 0:
            continue
        
        minDist = np.amin(dist[i])
        
        avg = np.zeros(len(groups))
        for j in range(len(groups)):
            avg[j] = avgDist(i, groups[j], dist)
        
        idfier = np.where(avg == np.min(avg))[0][0]
        gDist = dist[i, groups[idfier]]
        if np.min(gDist) < 0.3 * np.max(dist) and not np.min(gDist) > minDist:
            groups[idfier] = np.hstack((groups[idfier], i))
        else:
            groups.append(np.array([i]))
        active[i] = 0
    
    return groups


##########################################################################################
def groupPtgNodes(dist, dual_neighbours):
    """
    Simple decider to group ptg nodes based on distance

    Parameters
    ----------
    dist : (12,12)
        Distance matrix.

    Returns
    -------
    groups : (,)
        List of all groups.

    """
    groups = []
    active = np.ones(12, dtype=bool)
    
    #Step 0: Decide on where to start: on the nodes with minimal distance connections.
    distMin = np.amin(dist[dist > 0])
    current = np.any(dist == distMin, axis=1)
    
    #Step 0.5: Starting order, do nodes with max minimum connections first
    order = np.arange(0, 12, 1, dtype=int)
    ordered = False
        
    #Simple ordering algo
    while not ordered:
        ordered = True
        
        for k in range(11):
            if minCount(dist, order[(k+1)]) > minCount(dist, order[k]):
                order[(k+1)%12], order[k] = order[k], order[(k+1)]
                ordered = False
    
    
    #Step 1 : Get initial disjoint groups from ordered indices
    for index in order:
        #Take only nodes that we currently want to look at
        if not current[index] == True:
            continue
        
        group = nodeCluster(index, dual_neighbours, distMin)
        
        #Are all nb indices ungrouped? If not, the current index should be grouped with a grouped neighbour, if possible.
        if np.sum(active[group]) == group.shape[0]:
            #Now append the group and set the indices inactive
            groups.append(group)
        
            for x in group:
                active[x] = False
                
                
    #Step 2: Take care of leftover active indices
    groups = undecidedIndices(groups, dist, active)

    #Step 3: order the groups from center to outer nodes
    groups = sortGroups(groups, dist)
    
    return groups




############################ pocket region and patch initialization ###########################
def planRadii(group, dist):
    radii = np.zeros(group.shape[0])
    for i in range(group.shape[0]):
        
        #Single nodes get radius one
        if group.shape[0] == 1:
            radii[0] = 1
            continue
        
        #Define necessary arrays and parameters
        node = group[i]
        others = group[group != node]
        
        #Get nodes in minimal distance from our node
        distances = dist[node, others]
        minDist = np.amin(distances)
        
        nnb = np.where(distances == minDist)[0]
               
        #Define and calulate "compare", holding the index from the set of closest nodes (nnb) with the maximum radius
        #With that, determine the radius
        if np.sum(radii[nnb]) == 0:           
            radii[i] = np.ceil((minDist + 1)/2)
            
        else:
            compare = np.where(radii[nnb] == np.amax(radii[nnb]))[0][0]
            compare = nnb[compare]
            
            radii[i] = minDist + 1 -  radii[compare]

    radii = np.array(radii, dtype=int)
    return radii
        
# =============================================================================
# Index Identification
# =============================================================================

def idPatch(patchNB, graphNB, graphIndex):
    #Patch has a consistent indexing for all affected nodes. Therefore we design identifier with that length.
    nIndices = np.amax(patchNB) + 1
    
    #identifier array
    indexID = np.zeros(nIndices, dtype=int)
    
    #0th index in patch is always the graph index that we start from
    indexID[0] = graphIndex
    
    #get 1st circles
    graphCircle = nextLayer(graphNB, graphIndex)
    patchCircle = nextLayer(patchNB, 0)
    
    for i in range(patchCircle.shape[0]):
        indexID[patchCircle[i]] = graphCircle[i]
        
    #If extended patch grid, take 2nd layer
    if nIndices > 6:
        graphCircle = nextLayer(graphNB, graphCircle)
        patchCircle = nextLayer(patchNB, patchCircle)
        
        for i in range(patchCircle.shape[0]):
            indexID[patchCircle[i]] = graphCircle[i]
    
    return indexID

def findFace(face, pocket):
    indices = pocket.indexID
    isFound = False
    
    iExist = np.any(face[0] == indices) & np.any(face[1] == indices) & np.any(face[2] == indices)
    
    if iExist:
        pLayerFace = np.zeros(face.shape[0])
        for i in range(face.shape[0]):
            pLayerFace[i] = reverseID(face[i], indices)
        
        for pFace in pocket.faces:
            if np.any(pFace == pLayerFace[0]) & np.any(pFace == pLayerFace[1]) & np.any(pFace == pLayerFace[2]):
                isFound = True
                break
        
    if isFound:
        transform = np.zeros(3, dtype=int)
        for i in range(3):
            transform[i] = np.where(indices == face[i])[0][0]
    else:
        transform = None
        
    return isFound, transform
            
def commonFaces(pocket, patch, graphFaces):
    """
    Function to compute the overlapping faces between a pocket region and an adjacent patch. 
    Also orders the faces (center first) and computes rotation edge

    Parameters
    ----------
    pocket : PocketRegion() type object
        Pocket region.
    patch : Dual() type object
        Adjacent Patch.
    graphFaces : (,3)
        Graph Faces array that is the algorithm input.

    Returns
    -------
    overlap : (,2,3)
        Array of overlapping faces.1st index are the faces, 2nd index is for pocket/patch, 3rd index are the actual nodes
    rotEdge : (1,2) / NoneType
        If overlap array is of size (3,2,3), this returns the rotation edge of the patch (equivalent to the last DOF of that patch), 
        encoded in patch indices.

    """
    overlap = []
    
    for i in range(graphFaces.shape[0]):
        face = graphFaces[i]
        
        isFound1, transform1 = findFace(face, pocket)
        isFound2, transform2 = findFace(face, patch)
        
        if isFound1 and isFound2:
            overlap.append(np.array([transform1, transform2]))
    
    overlap = np.array(overlap)
    
    #If overlap has a shape[0] == 3, we are preparing a flat merge and need to choose the center face for positioning. Bring center face to pos 0.
    if overlap.shape[0] == 3:
        for i in range(overlap.shape[0]):
            face = overlap[i, 0]
            share = np.zeros(3, dtype=bool)
            
            share[0] = np.any(overlap[(i+1)%3] == face[0]) | np.any(overlap[(i+2)%3] == face[0])
            share[1] = np.any(overlap[(i+1)%3] == face[1]) | np.any(overlap[(i+2)%3] == face[1])
            share[2] = np.any(overlap[(i+1)%3] == face[2]) | np.any(overlap[(i+2)%3] == face[2])
            
            if np.sum(share) == 3 and not i == 0:
                a, b = overlap[i].copy(), overlap[0].copy()
                overlap[0], overlap[i] = a, b
                break
    
        #In this case, also calculate a rotation edge
        patchBDR = boundary(patch.faces)
        rotEdge = []
        
        for i in range(3):
            node = overlap[0, 1, i]
            if np.any(patchBDR == node):
                rotEdge.append(node)
    
    else:
        rotEdge = None
        
    return overlap, rotEdge

def commonNodes(patch1ID, patch2ID):

    common = []
    patchIndices = []
    for i in range(patch1ID.shape[0]):
        if np.any(patch2ID == patch1ID[i]):
            common.append(patch1ID[i])
            
            ids = np.array([i, np.where(patch2ID == patch1ID[i])[0][0]])
            patchIndices.append(ids)
    
    common = np.array(common)
    patchIndices = np.array(patchIndices)
    return common, patchIndices
             
def chooseConnectionIndices(indices, rotEdge):
    for i in range(indices.shape[1]):
        if not indices[i,0] in rotEdge:
            return indices[i]
    
    return None
# =============================================================================
# Merging     
# =============================================================================
def geometryFixingParam(group, active, positioned, gIndex, patches, pocket, overlap, rotEdge, selfFaces):
    #Check, if there is an overlap with one of the other faces
    common = np.array([])
    indices = np.array([])
    
    for patchJ in group[active]:
        i = np.where(group == patchJ)[0][0]
        
        if not i == gIndex:
            #common is the graph index, indices are the pocket and patch indices
            cCommon, cIndices = commonNodes(patches[gIndex].indexID, patches[i].indexID)
            
            if cCommon.shape[0] > 0:
                common, indices = cCommon, cIndices
                refGIndex = np.where(group == patchJ)[0][0]
                
                if positioned[refGIndex]:
                    break

    #INDICES:
        #gIndex, refGIndex for group indices
        #patchIndex, refPatchIndex for patch indices (which are the entries of groups).
    
    ###############Decide between edgelength, pointdist and symm_pointdist ###############
    
    #No common --> edgelength
    if common.shape[0] == 0:
        patchBDR = boundary(patches[gIndex].faces)
        pocketBDR = boundary(pocket.faces)
        
        for i in range(3):
            if  np.any(pocketBDR == overlap[1, 0,i]) & np.any(patchBDR == overlap[1,1,i]):
                comparePatch = overlap[1,1,i]
                comparePocket = overlap[1,0,i]
                break
        
        shiftIndex = np.where(pocketBDR == comparePocket)[0][0]
        comparePocket = pocketBDR[(shiftIndex + 1)%pocketBDR.shape[0]]
        
        #If that index appears in the overlapping face, it is the wrong one, as we want the first in bdr,
        #that is not overlapping. That we get by going the other direction
        if np.any(overlap[0,0] == comparePocket):
            comparePocket = pocketBDR[(shiftIndex - 1)%pocketBDR.shape[0]]
        
        inputObj = (gIndex, rotEdge, comparePocket, comparePatch)
        pocket.fixGeometry(inputObj, 'edgelength')
        
        #pocket.mergeFaces(pocket.patches[gIndex], overlap)
        positioned[gIndex] = True
    #Common --> pointdist or symm_pointdist
    else:
        
        indices = chooseConnectionIndices(indices, rotEdge)
        
        if positioned[refGIndex]:
            
            if type(indices) == None:
                raise SystemError("Could not find a suitable comparison index for current patch geometry fixing")
            
            inputObj = ([gIndex, refGIndex], rotEdge, indices[0], indices[1])
            pocket.fixGeometry(inputObj, 'pointdist')
            
            #pocket.mergeFaces(pocket.patches[gIndex], overlap)
            
            positioned[gIndex] = True
        else:
            
            #2 non-positioned patches --> symm_pointdist method. 
            #Compute overlap to cPatch and rotEdge
            overlap2, rotEdge2 = commonFaces(pocket, patches[refGIndex], selfFaces)
            
            #initial positioning of the new patch. Equates to step 1 of the 3-step patch merging process
            pocket.positionByFace(patches[refGIndex], overlap2[0])
            
            inputObj = ([gIndex,refGIndex], np.array([rotEdge, rotEdge2]), indices[0], indices[1])
            pocket.fixGeometry(inputObj, 'symm_pointdist')
            #pocket.mergeFaces(pocket.patches[refGIndex], overlap2)
            
            positioned[gIndex] = True
            positioned[refGIndex] = True
    return positioned


def transferRigidity(indexShift, patchRigid):
    pocketRigid = []
    for i in range(patchRigid.shape[0]):
        if not patchRigid[i]:
            continue
        
        pocketRigid.append(indexShift[i])
        
    pocketRigid = np.array(pocketRigid)
    
    return pocketRigid
# =============================================================================
# Pocket Region Growing
# =============================================================================
def selectCenter(dualNeighbours, graphInd, rigid):
    dist = rigidDist(dualNeighbours, rigid)
    indices = [i for i, x in enumerate(rigid) if x]
    indices = np.array(indices)
    
    rigidPtg = []
    for distIndex, i in enumerate(indices):
        if graphInd[i] < 12:
            rigidPtg.append(distIndex)
    
    centerNode = None
    centerAvgDist = np.inf
    
    for distIndex, i in enumerate(indices):
        avg = avgDist(distIndex, rigidPtg, dist)
        if avg < centerAvgDist:
            centerNode = i
            centerAvgDist = avg
            
    return centerNode

#Nikolai's code / CITE!
def calc_vertex_position(p0, p1, p2, l0, l1 , l2):
    """
    Calculates the position of an unknown vertex x, given three known vertices and the lengths between these and the unknown vertex
    """

    permute_coords = False
    permute_points = False
    
    # precondition to make resulting equations linear
    p1 = p1-p0
    p2 = p2-p0

    dot = p1.dot(p2)**2
    norms = p1.dot(p1)*p2.dot(p2)
    if dot == norms:
        # Points are colinear, Abort
        return
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    denom = x1*y2-y1*x2
    if denom == 0:
        # xy-projection of p1 and p2 are linearly dependent. Permute coordinates (ie, interchange columns)
        p1 = np.array((z1, x1, y1))
        p2 = np.array((z2, x2, y2))
        permute_coords = True

    if p1[0]==0:
        p1, p2 = p2, p1
        l1, l2 = l2, l1
        permute_points = True
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if permute_coords:
        pass
        # print("unpermuted coords")
        # print(f"x1={y1}, y1={z1}, z1={x1}")
    # Coefficients for the linear equations
    c1 = 0.5*(np.sum(p1**2) + l0**2-l1**2)
    c2 = 0.5*(np.sum(p2**2) + l0**2-l2**2)

    denom = x1*y2-y1*x2
    z2p = (x1*z2-x2*z1)/denom
    c2p = (x1*c2-x2*c1)/denom
    z1p = (z1-y1*z2p)/x1
    c1p = (c1-y1*c2p)/x1

    # Coefficients for quadratic constraints
    alpha = z1p*z1p + z2p*z2p + 1
    beta = -2*(z1p*c1p+z2p*c2p)
    gamma = c1p*c1p + c2p*c2p - l0*l0
    d = np.sqrt(beta*beta-4*alpha*gamma)

    # First test point
    z = (-beta+d)/(2*alpha)
    x = c1p - z1p*z
    y = c2p - z2p*z
    p_s1 = np.array((x, y, z))

    if permute_coords:
        p_s1 = np.array((y, z, x))

    # If it fails the test, we generate second point
    z = (-beta-d)/(2*alpha)
    x = c1p - z1p*z
    y = c2p - z2p*z
    p_s2 = np.array((x, y, z))
    if permute_coords:
        # Remember to permute back the coordinates!
        p_s2 = np.array((y, z, x))
    
    # Lastly, we translate the unknown point back with -p0.
    p_s1 = p_s1 + p0
    p_s2 = p_s2 + p0
    
    if np.linalg.norm(p_s2[:2]) > np.linalg.norm(p_s1[:2]):
        p = p_s2
    else:
        p = p_s1
        
    
    return p


def findCornerNodes(curLayer, lastLayer, dual_neighbours):
    
    #Define bool array that marks the corner nodes in curLayer
    isCorner = np.zeros(curLayer.shape[0], dtype=bool)
    
    for i in range(curLayer.shape[0]):
        neighbours = dual_neighbours[curLayer[i]]
        count = 0
        
        for nb in neighbours:
            if nb in lastLayer:
                count += 1
        
        if count == 1:
            isCorner[i] = True
    
    return isCorner


def firstTrue(arr):
    for i in range(arr.shape[0]):
        if arr[i] == True:
            return i
        
def calc_newCornerPos(curCorners, lastCorners, nodes):
    if lastCorners.shape[0] == 1 and curCorners.shape[0] == 5:
        newNodes = np.zeros((curCorners.shape[0], 3))
        
        for i in range(curCorners.shape[0]):
            newNodes[i,:] = nodes[curCorners[i]] + (nodes[curCorners[i]] - nodes[lastCorners[0]])
        
        return newNodes
    
    elif lastCorners.shape[0] >= 5 and lastCorners.shape[0] == curCorners.shape[0]:
        newNodes = np.zeros((curCorners.shape[0], 3))
        
        for i in range(curCorners.shape[0]):
            newNodes[i,:] = nodes[curCorners[i]] + (nodes[curCorners[i]] - nodes[lastCorners[i]])
        
        return newNodes
    
    else: 
        raise SystemError("Proportions of curCorners and lastCorners don't fit. Proportions either have to be equal or 5/1")
        
def calc_newSectorPos(cornerPos, l):
    
    #Number of sector nodes per sector
    s = l -1
    
    #Prepare array
    newSectorNodes = np.zeros((5,s,3))
    
    for i in range(5):
        k = cornerPos[(i+1)%5] - cornerPos[i]
        k = k / np.linalg.norm(k)
        
        for j in range(s):
            newSectorNodes[i,j,:] = cornerPos[i] + (j+1) * k
        
    return newSectorNodes

def orderNodeLayer(cornerNodes, sectorNodes):
    newNodes = np.vstack((cornerNodes[0,:], sectorNodes[0,:,:]))
    newCornersBit = np.hstack((np.array([True]), np.zeros(sectorNodes.shape[1], dtype=bool)))
    newCorners = newCornersBit.copy()
    
    
    for i in range(1, 5):
        newNodes = np.vstack((newNodes, cornerNodes[i,:], sectorNodes[i,:,:]))
        newCorners = np.hstack((newCorners, newCornersBit))
    
    return newNodes, newCorners

def facesBetweenLayers(innerLayer, outerLayer):
    faces = []
    
    l_i = innerLayer.shape[0]
    
    #Amount of nodes in a sector in the given layer (incl. one corner node)
    s_i = int(l_i / 5)
    s_o = s_i + 1
    
    for i in range(5):
        '''
        o_mask = np.zeros(outerLayer.shape[0], dtype=bool)
        i_mask = np.zeros(innerLayer.shape[0], dtype=bool)
        
        active = False
        index = 0
        while True:
            if index == i*s_i:
                active = True
            elif active and index == (i+1)*s_i+1:
                active = False
                break
            
            if active:
                i_mask[index%innerLayer.shape[0]] = True
            
            index += 1
        
        index = 0
        while True:
            if index == i*s_o:
                active = True
            elif active and index == (i+1)*s_o+1:
                active = False
                break
            
            if active:
                o_mask[index%outerLayer.shape[0]] = True
            
            index += 1
        
        outerSector = outerLayer[o_mask]
        innerSector = innerLayer[i_mask]
        '''
        if (i+1)*s_o+1 < outerLayer.shape[0]:
            outerSector = outerLayer[i*s_o:((i+1)*s_o+1)%outerLayer.shape[0]]
        else:
            outerSector = np.hstack((outerLayer[i*s_o:], outerLayer[:((i+1)*s_o+1)%outerLayer.shape[0]]))
            
        if (i+1)*s_i+1 < innerLayer.shape[0]:
            innerSector = innerLayer[i*s_i:((i+1)*s_i+1)%innerLayer.shape[0]]
        else:
            innerSector = np.hstack((innerLayer[i*s_i:], innerLayer[:((i+1)*s_i+1)%innerLayer.shape[0]]))
            
        
        for j in range(innerSector.shape[0] - 1):
            faces.append([innerSector[j], outerSector[j+1], innerSector[j+1]])
        
        for k in range(outerSector.shape[0] - 1):
            faces.append([outerSector[k], outerSector[k+1], innerSector[k]])
    
    faces = np.array(faces)
    
    return faces            
    
def layerNormal(layer, nodes):
    
    nodes = nodes[layer]
    
    center = plgCenter(nodes)
    
    radiusVectors = nodes - center
    normal = np.array([0,0,0])
    
    for i in range(radiusVectors.shape[0]):
        for j in range (radiusVectors.shape[0]):
            if i == j:
                continue
            elif np.isclose(np.linalg.norm(radiusVectors[i] - radiusVectors[j]), 0) or np.isclose(np.linalg.norm(radiusVectors[i] + radiusVectors[j]), 0):
                 continue
            
            partNormal = np.cross(radiusVectors[i], radiusVectors[j])
            partNormal /= np.linalg.norm(partNormal)
            
            if partNormal[2] >= 0:
                normal += partNormal
            else:
                normal -= partNormal
    
    normal /= np.linalg.norm(normal)
    
    return center, normal

def adjustPocketPosition(pocket, centerNode):
    layer = nextLayer(pocket.dual_neighbours, [centerNode])
    
    center, normal = layerNormal(layer, pocket.nodes)
    z = np.array([0,0,1])
    
    rot = np.cross(z, normal)
    angle = VectorAngle(z, normal)
    
    if rot > 1e-3:
        rotate(pocket, rot, center, -angle)
        
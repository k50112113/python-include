import numpy as np
from LocalFrameTransform import minimum_image, wrap

class SimulationBoxBinning:
    def __init__(self, lbox, nbin, coord=[]):
        self.lbox = lbox
        self.half_lbox = self.lbox/2
        self.nbin = nbin
        self.compute_neighbor_bin_list()
        if len(coord) > 0:
            self.update_coordinates(coord)
    
    def update_coordinates(self,coord):
        self.coord = coord
        self.do_binning()
    """
    def get_bin(self,id):
        return int((self.coord[id][0]+self.half_lbox)/self.lbox*self.nbin),\
               int((self.coord[id][1]+self.half_lbox)/self.lbox*self.nbin),\
               int((self.coord[id][2]+self.half_lbox)/self.lbox*self.nbin)
    """
    
    def do_binning(self):
        self.id_list = [[[[] for i in range(self.nbin)] for j in range(self.nbin)] for k in range(self.nbin)]
        self.ijk = (wrap(self.coord,self.lbox)/self.lbox*self.nbin).astype(int)
        for id in range(self.coord.shape[0]):
            self.id_list[self.ijk[id][0]][self.ijk[id][1]][self.ijk[id][2]].append(id)
    
    def compute_neighbor_bin_list(self):
        self.neighbor_bin_list = [[[[] for i in range(self.nbin)] for j in range(self.nbin)] for k in range(self.nbin)]
        fg = [-1,0,+1]
        if self.nbin > 1:
            for i in range(self.nbin):
                for j in range(self.nbin):
                    for k in range(self.nbin):
                        for di in fg:
                            for dj in fg:
                                for dk in fg:
                                    neighbor_i = (i+di)%self.nbin
                                    neighbor_j = (j+dj)%self.nbin
                                    neighbor_k = (k+dk)%self.nbin
                                    self.neighbor_bin_list[i][j][k].append([neighbor_i,neighbor_j,neighbor_k])
        else:
            self.neighbor_bin_list[0][0][0].append([0,0,0])

    def get_neighbor(self,id,cutoff):
        neighbor_id = []
        for neighbor_bin in self.neighbor_bin_list[self.ijk[id][0]][self.ijk[id][1]][self.ijk[id][2]]:
            neighbor_i = neighbor_bin[0]
            neighbor_j = neighbor_bin[1]
            neighbor_k = neighbor_bin[2]
            tmp_disp = minimum_image(self.coord[self.id_list[neighbor_i][neighbor_j][neighbor_k]] - self.coord[id],self.lbox)
            tmp_dist = np.linalg.norm(tmp_disp,axis=1)
            indices = np.where(tmp_dist < cutoff)[0].astype(int)
            for ii in indices:
                neighbor_id.append(self.id_list[neighbor_i][neighbor_j][neighbor_k][ii])
        return np.array(neighbor_id)
        
    def get_all_neighbor(self,cutoff):
        self.ngbr = [[] for i in range(self.coord.shape[0])]
        self.disp = np.zeros((self.coord.shape[0],self.coord.shape[0],3))
        self.dist = np.zeros((self.coord.shape[0],self.coord.shape[0]))
        
        for id in range(self.coord.shape[0]):
            for neighbor_bin in self.neighbor_bin_list[self.ijk[id][0]][self.ijk[id][1]][self.ijk[id][2]]:
                neighbor_i = neighbor_bin[0]
                neighbor_j = neighbor_bin[1]
                neighbor_k = neighbor_bin[2]      
                for neighbor_id in self.id_list[neighbor_i][neighbor_j][neighbor_k]:
                    if self.dist[id][neighbor_id] == 0 and neighbor_id!=id:
                        tmp_disp = minimum_image(self.coord[id] - self.coord[neighbor_id],self.lbox)
                        tmp_dist = np.sum(tmp_disp**2)**0.5
                        if tmp_dist < cutoff:
                            self.ngbr[id].append(neighbor_id)
                            self.ngbr[neighbor_id].append(id)
                            self.disp[id][neighbor_id] = tmp_disp
                            self.disp[neighbor_id][id] = -tmp_disp
                            self.dist[id][neighbor_id] = tmp_dist
                            self.dist[neighbor_id][id] = tmp_dist
                                    
        return self.ngbr,self.disp,self.dist
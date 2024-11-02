import numpy as np
import os

class LAMMPS_DATA:
    def __init__(self, filename="", atomstyle="full"):
        self.atoms            = 0
        self.atom_types       = 0
        self.bonds            = 0
        self.bond_types       = 0
        self.angles           = 0
        self.angle_types      = 0
        self.dihedrals        = 0
        self.dihedral_types   = 0
        self.impropers        = 0
        self.improper_types   = 0
        self.llo              = np.zeros(3)
        self.lhi              = np.zeros(3)
        self.tilt_factors     = np.zeros(3)
        self.lattice_vector   = np.zeros((3,3))
        
        self.molecule_id      = []
        self.type             = []
        self.charge           = []
        self.coord            = []
        self.vel              = []
        self.image            = []
        self.type_list        = []
        self.mass_list        = []
        
        self.fin = open(filename,"r")
        self.header = self.fin.readline()
        self.other_info = []
        self.fin.readline() #skip blank line
        for aline in self.fin:
            if "atoms" in aline:
                self.atoms = int(aline.strip().split()[0])
                self.molecule_id = np.zeros(self.atoms)
                self.type = np.zeros(self.atoms)
                self.charge = np.zeros(self.atoms)
                self.coord = np.zeros((self.atoms,3))
                self.vel = np.zeros((self.atoms,3))
                self.image = np.zeros((self.atoms,3))
            elif "atom types" in aline:
                self.atom_types = int(aline.strip().split()[0])
                self.type_list = np.zeros(self.atom_types)
                self.mass_list = np.zeros(self.atom_types)
            elif "bonds" in aline:
                self.bonds = int(aline.strip().split()[0])
            elif "bond types" in aline:
                self.bond_types = int(aline.strip().split()[0])
            elif "angles" in aline:
                self.angles = int(aline.strip().split()[0])
            elif "angle types" in aline:
                self.angle_types = int(aline.strip().split()[0])
            elif "dihedrals" in aline:
                self.dihedrals = int(aline.strip().split()[0])
            elif "dihedral types" in aline:
                self.dihedral_types = int(aline.strip().split()[0])
            elif "impropers" in aline:
                self.impropers = int(aline.strip().split()[0])
            elif "improper types" in aline:
                self.improper_types = int(aline.strip().split()[0])
            elif "xlo xhi" in aline:
                linelist = aline.strip().split()
                self.llo[0] = float(linelist[0])
                self.lhi[0] = float(linelist[1])
            elif "ylo yhi" in aline:
                linelist = aline.strip().split()
                self.llo[1] = float(linelist[0])
                self.lhi[1] = float(linelist[1])
            elif "zlo zhi" in aline:
                linelist = aline.strip().split()
                self.llo[2] = float(linelist[0])
                self.lhi[2] = float(linelist[1])
            elif "xy xz yz" in aline:
                linelist = aline.strip().split()
                self.tilt_factors = np.array([float(i) for i in linelist[:3]])
            elif "Masses" in aline:
                self.fin.readline()
                linelist = self.fin.readline().strip().split()
                id = 0
                while len(linelist) > 0:
                    self.type_list[id] = int(linelist[0])
                    self.mass_list[id] = float(linelist[1])
                    id += 1
                    linelist = self.fin.readline().strip().split()
            elif "Atoms" in aline:
                self.fin.readline()
                linelist = self.fin.readline().strip().split()
                if atomstyle == "full":
                    while len(linelist) > 0:
                        id = int(linelist[0]) - 1
                        self.molecule_id[id]  = int(linelist[1])
                        self.type[id]         = int(linelist[2])
                        self.charge     [id]  = float(linelist[3])
                        self.coord[id][0]     = float(linelist[4])
                        self.coord[id][1]     = float(linelist[5])
                        self.coord[id][2]     = float(linelist[6])
                        if len(linelist) > 7:
                            self.image[id][0] = float(linelist[7])
                            self.image[id][1] = float(linelist[8])
                            self.image[id][2] = float(linelist[9])
                        linelist = self.fin.readline().strip().split()
                elif atomstyle == "atomic":
                    while len(linelist) > 0:
                        id = int(linelist[0]) - 1
                        self.type[id]         = int(linelist[1])
                        self.coord[id][0]     = float(linelist[2])
                        self.coord[id][1]     = float(linelist[3])
                        self.coord[id][2]     = float(linelist[4])
                        if len(linelist) > 5:
                            self.image[id][0] = float(linelist[5])
                            self.image[id][1] = float(linelist[6])
                            self.image[id][2] = float(linelist[7])
                        linelist = self.fin.readline().strip().split()
            elif "Velocities" in aline:
                self.fin.readline()
                linelist = self.fin.readline().strip().split()
                index = 0
                while len(linelist) > 0:
                    id = int(linelist[0]) - 1
                    self.vel[id][0]      = float(linelist[1])
                    self.vel[id][1]      = float(linelist[2])
                    self.vel[id][2]      = float(linelist[3])
                    linelist = self.fin.readline().strip().split()
            else:
                self.other_info.append(aline)
        
        self.lattice_vector = np.array([[self.lhi[0]-self.llo[0], self.tilt_factors[0]   , self.tilt_factors[1]   ],\
                                        [0                      , self.lhi[1]-self.llo[1], self.tilt_factors[2]   ],\
                                        [0                      , 0                      , self.lhi[2]-self.llo[2]]]).T
        while len(self.other_info) > 0 and self.other_info[0].strip('\n') == "":
            del self.other_info[0]
    
    def get_volume(self):
        return np.linalg.det(self.lattice_vector)

    def reset_boundary(self,llo=[0.0,0.0,0.0]):
        self.coord -= self.llo
        self.lhi -= self.llo
        self.llo = np.array(llo)
        self.coord += self.llo
        self.lhi += self.llo

    def write_data(self,output_filename):
        with open(output_filename,"w") as fout:
            fout.write(self.header)
            fout.write('''
%d atoms
%d atom types
%d bonds
%d bond types
%d angles
%d angle types
%d dihedrals
%d dihedral types
%d impropers
%d improper types

%f %f xlo xhi
%f %f ylo yhi
%f %f zlo zhi
'''%(self.atoms,self.atom_types,\
     self.bonds,self.bond_types,\
     self.angles,self.angle_types,\
     self.dihedrals,self.dihedral_types,\
     self.impropers,self.improper_types,\
     self.llo[0],self.lhi[0],self.llo[1],self.lhi[1],self.llo[2],self.lhi[2]))
            if not all(self.tilt_factors == 0):
                fout.write('''%f %f %f xy xz yz\n'''%(self.tilt_factors[0], self.tilt_factors[1], self.tilt_factors[2]))
            fout.write('''
Masses

''')
            for i in range(self.type_list.shape[0]):
                fout.write("%d %f\n"%(self.type_list[i],self.mass_list[i]))
            fout.write('''
Atoms

''')        
            for i in range(self.atoms):
                fout.write("%d %d %d %f %f %f %f %d %d %d\n"%(i+1,self.molecule_id[i],self.type[i],self.charge[i],self.coord[i][0],self.coord[i][1],self.coord[i][2],self.image[i][0],self.image[i][1],self.image[i][2]))
            fout.write('''
Velocities

''')       
            for i in range(self.atoms):
                fout.write("%d %f %f %f\n"%(i+1,self.vel[i][0],self.vel[i][1],self.vel[i][2]))
            fout.write("\n")
            for others in self.other_info:
                fout.write(others)
                
class LAMMPS_ITR:
    def __init__(self, filename="", number_of_frames=False, additionalkey=[], copy=None):
        if number_of_frames == True:
            #compute number of frames
            fst = open(filename,"rb")
            for aline in fst:
                if b"ITEM: TIMESTEP" in aline:
                    step0 = int(fst.readline())
                    break
            for aline in fst:
                if b"ITEM: TIMESTEP" in aline:
                    f1_size = fst.tell()
                    step1 = int(fst.readline())
                    break
            fed = open(filename,"rb")
            fed.seek(0,os.SEEK_END)
            fed_size = fed.tell()
            fed.seek(fed_size-int(1.2*f1_size))
            for aline in fed:
                if b"ITEM: TIMESTEP" in aline:
                    stepn = int(fed.readline())
            # now_line = ""
            # while now_line.strip() != b'ITEM: TIMESTEP':
            #     now_pos-=1
            #     fed.seek(now_pos)
            #     new_char = fed.read(1)
            #     while new_char != b'\n' and now_pos > 0:
            #         now_pos-=1
            #         fed.seek(now_pos)
            #         new_char = fed.read(1)
            #     if now_pos > 0:
            #         fed.seek(now_pos+1)
            #     else:
            #         fed.seek(now_pos)
            #     now_line = fed.readline()
            #     print(now_line.strip())
            #stepn = int(fed.readline())
            self.number_of_frames = (stepn-step0)//(step1-step0)
            print(self.number_of_frames)
            #compute number of frames

        
        if copy:
            self.fin = open(copy.fin.name,"r")
            self.fin.seek(copy.fin_prestep_pointer)
        else:
            self.fin = open(filename,"r")
        self.fin_prestep_pointer = self.fin.tell()
        self.fin.readline()
        self.timestep = int(self.fin.readline())
        self.fin.readline()
        self.natom = int(self.fin.readline())

        self.llo = np.zeros(3)
        self.lhi = np.zeros(3)
        self.tilt_factors = np.zeros(3)
        boxtypeline = self.fin.readline()
        for k in range(3):
            linelist = self.fin.readline().strip().split()
            self.llo[k]          = float(linelist[0])
            self.lhi[k]          = float(linelist[1])
            if "xy xz yz" in boxtypeline: self.tilt_factors[k] = float(linelist[2])
        self.lattice_vector = np.array([[self.lhi[0]-self.llo[0], self.tilt_factors[0]   , self.tilt_factors[1]   ],\
                                        [0                      , self.lhi[1]-self.llo[1], self.tilt_factors[2]   ],\
                                        [0                      , 0                      , self.lhi[2]-self.llo[2]]])
        
        linelist = self.fin.readline().strip().split()
        self.id_index = None
        self.type_index = None
        self.element_index = None
        self.coord_index = None
        self.key_index = []
        self.additionalkey = additionalkey

        if "id" in linelist:
            self.id_index = linelist.index("id") - 2
        if "type" in linelist:
            self.type_index = linelist.index("type") - 2
        if "element" in linelist:
            self.element_index = linelist.index("element") - 2
        if "xu" in linelist:
            self.coord_index = linelist.index("xu") - 2
        elif "x" in linelist:
            self.coord_index = linelist.index("x") - 2
        elif "xs" in linelist:
            self.coord_index = linelist.index("xs") - 2
        elif "vx" in linelist:
            self.coord_index = linelist.index("vx") - 2

        for akey in self.additionalkey:
            self.key_index.append(linelist.index(akey) - 2)

        self.coord = np.zeros((self.natom,3))
        self.additionalkey_value = [[0 for j in range(self.natom)] for i in self.additionalkey]
        self.type = np.zeros(self.natom).astype(np.int32)
        self.element = ["" for i in range(self.natom)]
        for i in range(self.natom):
            linelist = self.fin.readline().strip().split()
            if self.id_index is not None:
                id = int(linelist[self.id_index]) - 1
            else:
                id = i
            if self.coord_index is not None:
                self.coord[id][0] = float(linelist[self.coord_index])
                self.coord[id][1] = float(linelist[self.coord_index+1])
                self.coord[id][2] = float(linelist[self.coord_index+2])
            if self.type_index is not None:
                self.type[id] = int(linelist[self.type_index])
            if self.element_index is not None:
                self.element[id] = linelist[self.element_index]
            for ikey in range(len(self.additionalkey)):
                self.additionalkey_value[ikey][id] = linelist[self.key_index[ikey]]

    def get_volume(self):
        return np.linalg.det(self.lattice_vector)

    def get_next(self,frame_interval=1):
        try:
            if frame_interval > 1:
                for f in range(frame_interval-1):
                    for i in range(9):
                        self.fin.readline()
                    for i in range(self.natom):
                        self.fin.readline()
            self.fin_prestep_pointer = self.fin.tell()
            self.fin.readline()
            self.timestep = int(self.fin.readline())
            self.fin.readline()
            self.natom = int(self.fin.readline())

            self.llo = np.zeros(3)
            self.lhi = np.zeros(3)
            self.tilt_factors = np.zeros(3)
            boxtypeline = self.fin.readline()
            for k in range(3):
                linelist = self.fin.readline().strip().split()
                self.llo[k]          = float(linelist[0])
                self.lhi[k]          = float(linelist[1])
                if "xy xz yz" in boxtypeline: self.tilt_factors[k] = float(linelist[2])
            self.lattice_vector = np.array([[self.lhi[0]-self.llo[0], self.tilt_factors[0]   , self.tilt_factors[1]   ],\
                                            [0                      , self.lhi[1]-self.llo[1], self.tilt_factors[2]   ],\
                                            [0                      , 0                      , self.lhi[2]-self.llo[2]]])
            linelist = self.fin.readline().strip().split()
            self.coord = np.zeros((self.natom,3))
            for i in range(self.natom):
                aline = self.fin.readline()
                linelist = aline.strip().split()
                if self.id_index is not None:
                    id = int(linelist[self.id_index]) - 1
                else:
                    id = i
                if self.coord_index is not None:
                    self.coord[id][0] = float(linelist[self.coord_index])
                    self.coord[id][1] = float(linelist[self.coord_index+1])
                    self.coord[id][2] = float(linelist[self.coord_index+2])
                for ikey in range(len(self.additionalkey)):
                    self.additionalkey_value[ikey][id] = linelist[self.key_index[ikey]]
            return True
        except:
            return False

class LAMMPS_dump_writer:
    def __init__(self, filename, is_triclinic = False, outputkeys = ["id", "type", "x y z"]):
        self.fout = open(filename, "w")
        self.is_triclinic = is_triclinic
        self.outputkeys = outputkeys
    
    def dump_frame(self, timestep, natom, lattice_vector, data):
        self.fout.write('''ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n'''%(timestep, natom))
        if self.is_triclinic == True:
            boxtypeline = "xy xz yz"
            tilt_factors = ["%e"%lattice_vector[0][1],"%e"%lattice_vector[0][2], "%e"%lattice_vector[1][2]]
        else:
            boxtypeline = "pp pp pp"
            tilt_factors = ["","",""]
        self.fout.write('''ITEM: BOX BOUNDS %s\n%d %e %s\n%d %e %s\n%d %e %s\n'''%(boxtypeline, 0, lattice_vector[0][0], tilt_factors[0],\
                                                                                                0, lattice_vector[1][1], tilt_factors[1],\
                                                                                                0, lattice_vector[2][2], tilt_factors[2]))
        self.fout.write('''ITEM: ATOMS ''')
        for a_key in self.outputkeys:
            self.fout.write('''%s '''%(a_key))
        self.fout.write('''\n''')
        for i in range(natom):
            for a_key in self.outputkeys:
                if type(np.array([1])) == type(data[a_key][i]) or type([1]) == type(data[a_key][i]):
                    for j in range(len(data[a_key][i])):
                        self.fout.write('''%e '''%(data[a_key][i][j]))
                else:
                    self.fout.write('''%s '''%(data[a_key][i]))
            self.fout.write('''\n''')

    def close(self):
        self.fout.close()

class XYZ_ITR:
    def __init__(self, filename):
        self.fin = open(filename,"r")
        self.natom = int(self.fin.readline())
        self.fin.readline()
        self.coord = np.zeros((self.natom,3))
        self.type = ["" for i in range(self.natom)]
        for i in range(self.natom):
            linelist = self.fin.readline().strip().split()
            self.coord[i][0] = float(linelist[1])
            self.coord[i][1] = float(linelist[2])
            self.coord[i][2] = float(linelist[3])
            self.type[i] = linelist[0]
    
    def get_next(self,frame_interval=1):
        try:
            for f in range(frame_interval):
                self.natom = int(self.fin.readline())
                self.fin.readline()
                self.coord = np.zeros((self.natom,3))
                self.type = ["" for i in range(self.natom)]
                for i in range(self.natom):
                    linelist = self.fin.readline().strip().split()
                    self.coord[i][0] = float(linelist[1])
                    self.coord[i][1] = float(linelist[2])
                    self.coord[i][2] = float(linelist[3])
                    self.type[i] = linelist[0]
            return True
        except:
            return False    

def read_lammps_output(filename,keywords): #first column must be Step
    fin = open(filename,"r")
    data = []
    step = []
    aline = fin.readline()
    keywords_list = keywords.strip().split()
    while keywords not in aline:
        aline = fin.readline()
    linelist = aline.strip().split()
    keywords_start_index = linelist.index(keywords_list[0])
    aline = fin.readline()
    while "Loop time" not in aline:
        linelist = aline.strip().split()
        tmpdata = [float(i) for i in linelist[keywords_start_index:keywords_start_index+len(keywords_list)]]
        data.append(tmpdata)
        step.append(int(linelist[0]))
        aline = fin.readline()
    fin.close()        
    return np.array(data),np.array(step)

def read_lammps_data(filename,start_frame,end_frame,molecule=False):
    coord = []
    type = []
    with open(filename,"r") as fin:
        nowframe = 0
        for aline in fin:
            if "TIMESTEP" in aline:
                nowtimestep = int(fin.readline())
                nowframe += 1
                if nowframe > end_frame:
                    break
                elif nowframe >= start_frame:
                    fin.readline()
                    natom = int(fin.readline())
                    fin.readline()
                    linelist = fin.readline().strip().split()
                    lbox = float(linelist[1])-float(linelist[0])
                    fin.readline()
                    fin.readline()
                    newline = fin.readline()
                    
                    if len(coord) == 0:
                        linelist = newline.strip().split()
                        id_index = linelist.index("id") - 2
                        type_index = linelist.index("type") - 2
                        if molecule == True:
                            if "mol" in linelist:
                                mol_index = linelist.index("mol") - 2
                            else:
                                print("Molecuar id not found.")
                                exit()
                        if "xu" in linelist:
                            coord_index = linelist.index("xu") - 2
                        elif "x" in linelist:
                            coord_index = linelist.index("x") - 2
                        
                        firstdata = []
                        for i in range(natom):
                            linelist = fin.readline().strip().split()
                            if molecule == True:
                                firstdata.append([int(linelist[id_index]),int(linelist[type_index]),[float(i) for i in linelist[coord_index:]],int(linelist[mol_index])])
                            else:
                                firstdata.append([int(linelist[id_index]),int(linelist[type_index]),[float(i) for i in linelist[coord_index:]]])
                        firstdata.sort(key=lambda x: x[0])
                        coord.append([[] for i in range(len(firstdata))])
                        type = [0 for i in range(len(firstdata))]
                        mol = [0 for i in range(len(firstdata))]
                        for i in range(natom):
                            type[i] = firstdata[i][1]
                            coord[len(coord)-1][i] = firstdata[i][2]
                            if molecule == True:
                                mol[i] = firstdata[i][3]
                    else:
                        coord.append([[] for i in range(len(firstdata))])
                        for i in range(natom):
                            linelist = fin.readline().strip().split()
                            coord[len(coord)-1][int(linelist[id_index])-1] = [float(i) for i in linelist[coord_index:]]
    
    coord=np.array(coord)
    print("Total number of frames: %d"%(len(coord)))
    return coord,type,natom,lbox,len(coord),mol

class VASP_DATA:
    def __init__(self, filename=""):
        self.fin = open(filename,"r")
        self.headline = self.fin.readline()
        self.cellscale = float(self.fin.readline().strip())
        self.lattice_vector = []
        for j in range(3):
            self.lattice_vector.append([float(i) for i in self.fin.readline().strip().split()])
        self.lattice_vector = np.array(self.lattice_vector)
        self.atom_type_list = [i for i in self.fin.readline().strip().split()]
        self.atom_num_list = np.array([int(i) for i in self.fin.readline().strip().split()])
        self.atoms = np.sum(self.atom_num_list)
        self.coord = np.zeros((self.atoms,3))
        self.type = []
        for a_atom_type_index, a_atom_type in enumerate(self.atom_type_list):
            self.type += [a_atom_type]*self.atom_num_list[a_atom_type_index]

        self.coord_type = self.fin.readline().strip()
        
        for j in range(self.atoms):
            linelist = self.fin.readline().strip().split()
            self.coord[j][0] = float(linelist[0])
            self.coord[j][1] = float(linelist[1])
            self.coord[j][2] = float(linelist[2])

    def write_data(self,filename):
        with open(filename,"w") as fout:
            fout.write(self.headline)
            fout.write("%s"%(self.cellscale))
            fout.write("\n")
            for j in range(3):
                for i in range(3):
                    fout.write("%f "%(self.lattice_vector[j][i]))
                fout.write("\n")
            for a_atom_type in self.atom_type_list:
                fout.write("%s "%(a_atom_type))
            fout.write("\n")
            for a_atom_num in self.atom_num_list:
                fout.write("%d "%(a_atom_num))
            fout.write("\n")
            fout.write(self.coord_type)
            fout.write("\n")
            for j in range(self.atoms):
                fout.write("%f %f %f\n"%(self.coord[j][0],self.coord[j][1],self.coord[j][2]))
           
class VASP_ITR:
    def __init__(self, filename, number_of_frames=False):
        self.fin = open(filename,"r")
        self.headline = self.fin.readline()
        self.cellscale = float(self.fin.readline().strip())
        self.lattice_vector = []
        for j in range(3):
            self.lattice_vector.append([float(i) for i in self.fin.readline().strip().split()])
        self.lattice_vector = np.array(self.lattice_vector)
        self.atom_type_list = [i for i in self.fin.readline().strip().split()]
        self.atom_num_list = np.array([int(i) for i in self.fin.readline().strip().split()])
        self.natom = np.sum(self.atom_num_list)
        self.coord = np.zeros((self.natom,3))
        self.type = []
        self.timestep = 0
        for a_atom_type_index, a_atom_type in enumerate(self.atom_type_list):
            self.type += [a_atom_type]*self.atom_num_list[a_atom_type_index]

        self.coord_type = self.fin.readline().strip()
        
        for j in range(self.natom):
            linelist = self.fin.readline().strip().split()
            self.coord[j][0] = float(linelist[0])
            self.coord[j][1] = float(linelist[1])
            self.coord[j][2] = float(linelist[2])

    def get_next(self,frame_interval=1):
        try:
            if frame_interval > 1:
                for f in range(frame_interval-1):
                    self.fin.readline()
                    for i in range(self.natom):
                        self.fin.readline()
            aline = self.fin.readline()
            self.timestep = int(aline.strip().split()[-1])
            self.coord = np.zeros((self.natom,3))
            for i in range(self.natom):
                aline = self.fin.readline()
                linelist = aline.strip().split()
                self.coord[i][0] = float(linelist[0])
                self.coord[i][1] = float(linelist[1])
                self.coord[i][2] = float(linelist[2])
            return True
        except:
            return False

def read_vasp_traj(filename,start_frame=0,end_frame=-1,normal_output=True):
    coord = []
    sys_info = []
    try:
        with open(filename,"r") as xdatcar_in:
            newline = xdatcar_in.readline() #system name
            sys_info.append(newline)
            
            newline = xdatcar_in.readline() #lattice factor
            sys_info.append(newline)
            
            newline = xdatcar_in.readline() #x vector
            sys_info.append(newline)
            lx = float(newline.strip().split()[0])
            
            newline = xdatcar_in.readline() #y vector
            sys_info.append(newline)
            ly = float(newline.strip().split()[1])
            
            newline = xdatcar_in.readline() #z vector
            sys_info.append(newline)
            lz = float(newline.strip().split()[2])
            
            newline = xdatcar_in.readline() #atom type
            sys_info.append(newline)
            type_list = newline.strip().split()
            
            newline = xdatcar_in.readline() #number of atoms
            sys_info.append(newline)
            atom_num_list = [int(i) for i in newline.strip().split()]
            nowframe = 0
            for aline in xdatcar_in:
                if "Direct configuration" in aline:
                    nowframe += 1
                    if nowframe > end_frame and end_frame != -1:
                        break
                    elif nowframe >= start_frame:
                        coord.append([])
                    
                else:
                    if nowframe >= start_frame:
                        linelist = aline.strip().split()
                        coord[len(coord)-1].append([float(i) for i in linelist])
    except IOError:
        print("File Not Found.")
        exit()
        
    lbox = np.array([lx,ly,lz])
    
    type = []
    for atom_index, atom in enumerate(type_list):
        type += [atom]*atom_num_list[atom_index]
    natom = sum(atom_num_list)    
    coord=np.array(coord)*lbox
    print("Total number of frames: %d"%(coord.shape[0]))
    if normal_output:
        return coord,type,natom,lbox,coord.shape[0]
    else:
        return coord,type_list,atom_num_list,lbox,coord.shape[0],sys_info

def read_vasp_output(filename):
    temp = []
    press = []
    coord = []
    force = []
    virial = []
    energy = []
    lattice_vector = []

    with open(filename,"r") as outcar_in:
        for aline in outcar_in:
            if "(temperature" in aline:
                tmp = aline[aline.index("(temperature")+len("(temperature"):]
                tmp = tmp.split()[0]
                temp.append(float(tmp))
            elif "total pressure  =" in aline:
                press.append(float(aline.strip().split()[3]))
            elif "POSITION" in aline and "TOTAL-FORCE (eV/Angst)" in aline:
                coord.append([])
                force.append([])
                outcar_in.readline() #-------...
                newline = outcar_in.readline()
                while "---" not in newline:
                    linelist = newline.strip().split()
                    coord[-1].append(linelist[0:3])
                    force[-1].append(linelist[3:6])
                    newline = outcar_in.readline()
            elif "in kB" in aline:
                #2  3  4  5  6  7
                #XX YY ZZ XY YZ ZX
                #2  5  7  5  3  6  7  6  4
                #xx xy xz yx yy yz zx zy zz
                linelist = aline.strip().split()
                virial.append([str(float(linelist[i])*1000) for i in [2,5,7,5,3,6,7,6,4]])
            elif "ion-electron   TOTEN  =" in aline: 
                linelist = aline.strip().split()
                energy.append(linelist[4])
            elif "direct lattice vectors" in aline:
                if len(press) > 0:
                    lattice_vector.append([])
                    newline = outcar_in.readline()
                    linelist = newline.strip().split()
                    lattice_vector[-1].append(linelist[0:3])
                    newline = outcar_in.readline()
                    linelist = newline.strip().split()
                    lattice_vector[-1].append(linelist[0:3])
                    newline = outcar_in.readline()
                    linelist = newline.strip().split()
                    lattice_vector[-1].append(linelist[0:3])
            # elif"total energy   ETOTAL =" in aline:
            #     linelist = aline.strip().split()
            #     energy.append(linelist[4])
                
    return temp, press, coord, force, energy, virial, lattice_vector
   

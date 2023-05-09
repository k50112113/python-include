import numpy as np
from queue import Queue
import ReadMD as RM
from scipy.spatial.transform import Rotation as R

class MoleculeTopology:
    def __init__(self, top_file,pdb_file):
        with open(top_file,"r") as fin:
            self.n_types = int(fin.readline())
            self.type_list_ = []
            self.mass_list_ = []
            for i in range(self.n_types):
                linelist = fin.readline().strip().split()
                self.type_list_.append(linelist[0])
                self.mass_list_.append(float(linelist[1]))
            self.n_atoms = int(fin.readline())
            self.type_ = []
            for i in range(self.n_atoms):
                self.type_.append(fin.readline().strip())
            self.n_bonds = int(fin.readline())
            self.bond_list_ = []
            self.n_bond_types = 0
            self.bond_type_list_ = []
            self.bond_type_dict = {}
            self.bond_map = np.zeros((self.n_atoms,self.n_atoms))
            for i in range(self.n_bonds):
                linelist = fin.readline().strip().split()
                id_1 = int(linelist[0])-1
                id_2 = int(linelist[1])-1
                self.bond_map[id_1][id_2] = 1
                self.bond_map[id_2][id_1] = 1
                type_chain = [self.type_[id_1],self.type_[id_2]]
                if self.bond_type_dict.get(self.chain_to_key(type_chain)) == None:
                    self.bond_type_list_.append(type_chain[:])
                    self.bond_type_dict[self.chain_to_key(type_chain)] = self.n_bond_types+1
                    self.bond_type_dict[self.chain_to_key(type_chain[::-1])] = self.n_bond_types+1
                    self.n_bond_types += 1
                self.bond_list_.append([id_1,id_2])
            self.angle_dict = {}
            self.dihedral_dict = {}

            self.angle_list_ = []
            self.n_angle_types = 0
            self.angle_type_list_ = []
            self.angle_type_dict = {}
            self.dihedral_list_ = []
            self.n_dihedral_types = 0
            self.dihedral_type_list_ = []
            self.dihedral_type_dict = {}
            q = Queue()
            for i in range(self.n_atoms):
                q.put([i])
            while q.empty()==False:
                now_chain = q.get()
                for new_id in range(self.n_atoms):
                    if self.bond_map[now_chain[-1]][new_id] == 1:
                        if len(now_chain) < 2 or now_chain[-2] != new_id:
                            new_chain = now_chain + [new_id]
                            if len(new_chain) <= 3:
                                q.put(new_chain[:])
                            type_chain = [self.type_[k] for k in new_chain]
                            if len(new_chain) == 3:
                                if self.angle_dict.get(self.chain_to_key(new_chain)) == None:
                                    self.angle_dict[self.chain_to_key(new_chain)] = True
                                    self.angle_dict[self.chain_to_key(new_chain[::-1])] = True
                                    self.angle_list_.append(new_chain[:])
                                    if self.angle_type_dict.get(self.chain_to_key(type_chain)) == None:
                                        self.angle_type_list_.append(type_chain[:])
                                        self.angle_type_dict[self.chain_to_key(type_chain)] = self.n_angle_types+1
                                        self.angle_type_dict[self.chain_to_key(type_chain[::-1])] = self.n_angle_types+1
                                        self.n_angle_types += 1
                            elif len(new_chain) == 4:
                                if self.dihedral_dict.get(self.chain_to_key(new_chain)) == None:
                                    self.dihedral_dict[self.chain_to_key(new_chain)] = True
                                    self.dihedral_dict[self.chain_to_key(new_chain[::-1])] = True
                                    self.dihedral_list_.append(new_chain[:])
                                    if self.dihedral_type_dict.get(self.chain_to_key(type_chain)) == None:
                                        self.dihedral_type_list_.append(type_chain[:])
                                        self.dihedral_type_dict[self.chain_to_key(type_chain)] = self.n_dihedral_types+1
                                        self.dihedral_type_dict[self.chain_to_key(type_chain[::-1])] = self.n_dihedral_types+1
                                        self.n_dihedral_types += 1

        self.coord = np.zeros((self.n_atoms,3))
        with open(pdb_file,"r") as fin:
            fin.readline()
            for i in range(self.n_atoms):
                aline = fin.readline()
                linelist = aline.strip().split()
                self.coord[i][0] = float(linelist[-6])
                self.coord[i][1] = float(linelist[-5])
                self.coord[i][2] = float(linelist[-4])

    def output_lt_file(self,outputfilename="tmp.lt",name="TMP"):
        print(self.n_bond_types)
        for a_bond_type in self.bond_type_list_:
            for a_type in a_bond_type:
                print(a_type, end='\t')
            print("")
        print(self.n_angle_types)
        for a_angle_type in self.angle_type_list_:
            for a_type in a_angle_type:
                print(a_type, end='\t')
            print("")
        print(self.n_dihedral_types)
        for a_dihedral_type in self.dihedral_type_list_:
            for a_type in a_dihedral_type:
                print(a_type, end='\t')
            print("")
        fout = open(outputfilename,'w')
        fout.write("%s{\n"%(name))

        fout.write("""  write("Data Atoms") { \n""")
        for index in range(self.n_atoms):
            fout.write("""    $atom:%s  $mol:. @atom:%s   %s  %s  %s  %s   # %s\n"""%(\
            str(index+1).ljust(4),\
            str(self.type_list_.index(self.type_[index])+1).ljust(4),\
            0.0,\
            str(self.coord[index][0]).ljust(8),\
            str(self.coord[index][1]).ljust(8),\
            str(self.coord[index][2]).ljust(8),\
            str(self.type_[index]).ljust(3)))
        fout.write("  }\n")
        fout.write("""  write_once("Data Masses") { \n""")
        for index in range(self.n_types):
            fout.write("""    @atom:%s %s # %s\n"""%(\
            str(index+1).ljust(4),\
            str(self.mass_list_[index]).ljust(8),\
            str(self.type_list_[index]).ljust(3)))
        fout.write("  }\n")
        fout.write("""  write("Data Bonds") { \n""")
        for index in range(len(self.bond_list_)):
            type_chain = [self.type_[k] for k in self.bond_list_[index]]
            fout.write("""    $bond:%s  @bond:%s $atom:%s  $atom:%s    # %s %s\n"""%(\
            str(index+1).ljust(4),\
            str(self.bond_type_dict[self.chain_to_key(type_chain)]).ljust(4),\
            str(self.bond_list_[index][0]+1).ljust(4),\
            str(self.bond_list_[index][1]+1).ljust(4),\
            str(type_chain[0]).ljust(4),\
            str(type_chain[1]).ljust(4)))

        fout.write("  }\n")
        fout.write("""  write("Data Angles") { \n""")
        for index in range(len(self.angle_list_)):
            type_chain = [self.type_[k] for k in self.angle_list_[index]]
            fout.write("""    $angle:%s  @angle:%s $atom:%s  $atom:%s  $atom:%s  # %s %s %s\n"""%(\
            str(index+1).ljust(4),\
            str(self.angle_type_dict[self.chain_to_key(type_chain)]).ljust(4),\
            str(self.angle_list_[index][0]+1).ljust(4),\
            str(self.angle_list_[index][1]+1).ljust(4),\
            str(self.angle_list_[index][2]+1).ljust(4),\
            str(type_chain[0]).ljust(4),\
            str(type_chain[1]).ljust(4),\
            str(type_chain[2]).ljust(4)))
        fout.write("  }\n")
        fout.write("""  write("Data Dihedrals") { \n""")
        for index in range(len(self.dihedral_list_)):
            type_chain = [self.type_[k] for k in self.dihedral_list_[index]]
            fout.write("""    $dihedral:%s  @dihedral:%s    $atom:%s    $atom:%s   $atom:%s  $atom:%s  # %s %s %s %s\n"""%(\
            str(index+1).ljust(4),\
            str(self.dihedral_type_dict[self.chain_to_key(type_chain)]).ljust(4),\
            str(self.dihedral_list_[index][0]+1).ljust(4),\
            str(self.dihedral_list_[index][1]+1).ljust(4),\
            str(self.dihedral_list_[index][2]+1).ljust(4),\
            str(self.dihedral_list_[index][3]+1).ljust(4),\
            str(type_chain[0]).ljust(4),\
            str(type_chain[1]).ljust(4),\
            str(type_chain[2]).ljust(4),\
            str(type_chain[3]).ljust(4)))
        fout.write("  }\n")
        fout.write("}\n")
        fout.close()

    def chain_to_key(self,chain):
        key = "%s"%(chain[0])
        for a in chain[1:]:
            key += "_%s"%(a)
        return key
    def key_to_chain(self,key):
        chain = [int(i) for i in key.split('_')]
        return chain

                                

class molecule:
    def __init__(self, coord=[], top_file=""):
        self.coord = coord
        with open(top_file,"r") as fin:
            self.atom_types = int(fin.readline())
            self.type_list = []
            self.mass_list = []
            for i in range(self.atom_types):
                linelist = fin.readline().strip().split()
                self.type_list.append(linelist[0])
                self.mass_list.append(float(linelist[1]))
            self.atoms = int(fin.readline())
            self.bond_map = np.zeros((self.atoms,self.atoms))
            self.bond_list = []
            self.type = []
            for i in range(self.atoms):
                self.type.append(fin.readline().strip())
            self.bond_type_string_list = []
            self.angle_type_string_list = []
            self.dihedral_type_string_list = []
            self.bond_type_list = []
            self.angle_type_list = []
            self.dihedral_type_list = []
            n = int(fin.readline().strip().split().pop())
            for i in range(n):
                linelist = fin.readline().strip().split()
                self.bond_type_list.append(int(linelist.pop()))
                self.bond_type_string_list.append(linelist[:])
            n = int(fin.readline().strip().split().pop())
            for i in range(n):
                linelist = fin.readline().strip().split()
                self.angle_type_list.append(int(linelist.pop()))
                self.angle_type_string_list.append(linelist[:])
            n = int(fin.readline().strip().split().pop())
            for i in range(n):
                linelist = fin.readline().strip().split()
                self.dihedral_type_list.append(int(linelist.pop()))
                self.dihedral_type_string_list.append(linelist[:])
            fin.readline()
            for aline in fin:
                linelist = aline.strip().split()
                atom1 = int(linelist[0])
                atom2 = int(linelist[1])
                self.bond_map[atom1-1][atom2-1] = 1
                self.bond_map[atom2-1][atom1-1] = 1
                self.bond_list.append([atom1,atom2])
        self.angle_list = []
        self.dihedral_list = []
        q = Queue()
        for i in range(1,self.atoms+1):
            q.put([i])
        while q.empty() == False:
            seq = q.get()
            seq_end = seq[len(seq)-1]
            for i in range(1,self.atoms+1):
                if i not in seq and self.bond_map[i-1][seq_end-1] == 1:
                    new_seq = seq+[i]
                    if len(new_seq) == 4:
                        passs = True
                        for a_seq in self.dihedral_list:
                            if np.linalg.norm(np.array(a_seq[::-1])-np.array(new_seq)) == 0:
                                passs = False
                                break
                        if passs == True:
                            self.dihedral_list.append(new_seq)
                    else:
                        if len(new_seq) == 3:
                            passs = True
                            for a_seq in self.angle_list:
                                if np.linalg.norm(np.array(a_seq[::-1])-np.array(new_seq)) == 0:
                                    passs = False
                                    break
                            if passs == True:
                                self.angle_list.append(new_seq)
                        q.put(new_seq)
        
        self.bond_type = []
        self.angle_type = []
        self.dihedral_type = []
        
        ##############################################################################################
        for a_bond in self.bond_list:
            tmp_bond_type_string = []
            for a_atom_id in a_bond:
                tmp_bond_type_string.append(self.type[a_atom_id-1])
            found_bond_type = 0
            for bond_index, a_bond_type_string_list in enumerate(self.bond_type_string_list):
                if tmp_bond_type_string == a_bond_type_string_list or tmp_bond_type_string == a_bond_type_string_list[::-1]:
                    self.bond_type.append(self.bond_type_list[bond_index])
                    found_bond_type = 1
                    break
            if found_bond_type == 0:
                self.bond_type_string_list.append(tmp_bond_type_string[:])
                if len(self.bond_type_list) > 0:
                    self.bond_type_list.append(max(self.bond_type_list)+1)
                    self.bond_type.append(self.bond_type_list[len(self.bond_type_list)-1])
                else:
                    self.bond_type_list.append(1)
                    self.bond_type.append(1)
        ##############################################################################################
        for a_angle in self.angle_list:
            tmp_angle_type_string = []
            for a_atom_id in a_angle:
                tmp_angle_type_string.append(self.type[a_atom_id-1])
            found_angle_type = 0
            for angle_index, a_angle_type_string_list in enumerate(self.angle_type_string_list):
                if tmp_angle_type_string == a_angle_type_string_list or tmp_angle_type_string == a_angle_type_string_list[::-1]:
                    self.angle_type.append(self.angle_type_list[angle_index])
                    found_angle_type = 1
                    break
            if found_angle_type == 0:
                self.angle_type_string_list.append(tmp_angle_type_string[:])
                if len(self.angle_type_list) > 0:
                    self.angle_type_list.append(max(self.angle_type_list)+1)
                    self.angle_type.append(self.angle_type_list[len(self.angle_type_list)-1])
                else:
                    self.angle_type_list.append(1)
                    self.angle_type.append(1)
        ##############################################################################################
        for a_dihedral in self.dihedral_list:
            tmp_dihedral_type_string = []
            for a_atom_id in a_dihedral:
                tmp_dihedral_type_string.append(self.type[a_atom_id-1])
            found_dihedral_type = 0
            for dihedral_index, a_dihedral_type_string_list in enumerate(self.dihedral_type_string_list):
                if tmp_dihedral_type_string == a_dihedral_type_string_list or tmp_dihedral_type_string == a_dihedral_type_string_list[::-1]:
                    self.dihedral_type.append(self.dihedral_type_list[dihedral_index])
                    found_dihedral_type = 1
                    break
            if found_dihedral_type == 0:
                self.dihedral_type_string_list.append(tmp_dihedral_type_string[:])
                if len(self.dihedral_type_list) > 0:
                    self.dihedral_type_list.append(max(self.dihedral_type_list)+1)
                    self.dihedral_type.append(self.dihedral_type_list[len(self.dihedral_type_list)-1])
                else:
                    self.dihedral_type_list.append(1)
                    self.dihedral_type.append(1)
        ##############################################################################################
        
    def make_lammps_data(self,filename="data"):
        data = RM.LAMMPS_DATA()
        data.header           = "Molecule made by MoleculeConstructor\n"
        data.atoms            = self.atoms
        data.atom_types       = self.atom_types
        data.bonds            = len(self.bond_list)
        data.bond_types       = len(np.unique(self.bond_type_list))
        data.angles           = len(self.angle_list)
        data.angle_types      = len(np.unique(self.angle_type_list))
        data.dihedrals        = len(self.dihedral_list)
        data.dihedral_types   = len(np.unique(self.dihedral_type_list))
        data.impropers        = 0
        data.improper_types   = 0
        data.coord            = self.coord
        #data.coord            = np.ones((self.atoms,3))
        data.vel              = np.zeros((self.atoms,3))
        data.image            = np.zeros((self.atoms,3))
        
        data.molecule_id      = np.ones(self.atoms)
        data.type_list        = np.array([i+1 for i in range(self.atom_types)])
        data.mass_list        = np.array(self.mass_list)
        data.type             = np.array([data.type_list[self.type_list.index(i)] for i in self.type])
        data.charge           = np.zeros(self.atoms)
        
        data.lx0              = data.coord.min()-abs(data.coord.min())*10
        data.lx1              = data.coord.max()+abs(data.coord.max())*10
        data.ly0              = data.lx0
        data.ly1              = data.lx1  
        data.lz0              = data.lx0
        data.lz1              = data.lx1  
        
        
        data.other_info       = []
        data.other_info.append("Bonds\n\n")
        for i in range(len(self.bond_list)):
            tmp = "%d %d "%(i+1,self.bond_type[i])
            for j in self.bond_list[i]:
                tmp += "%d "%(j)
            data.other_info.append(tmp+'\n')
        data.other_info.append("\nAngles\n\n")
        for i in range(len(self.angle_list)):
            tmp = "%d %d "%(i+1,self.angle_type[i])
            for j in self.angle_list[i]:
                tmp += "%d "%(j)
            data.other_info.append(tmp+'\n')
        data.other_info.append("\nDihedrals\n\n")
        for i in range(len(self.dihedral_list)):
            tmp = "%d %d "%(i+1,self.dihedral_type[i])
            for j in self.dihedral_list[i]:
                tmp += "%d "%(j)
            data.other_info.append(tmp+'\n')
        data.write_data(filename)
                    

    def adjust_h(self,coord=[]):
        if len(coord)>0:
            self.coord = coord
            
        angle_CCH = 109.4712*np.pi/180
        angle_HCH = np.pi*2/3
        length_CH = 1.09   
        for c1_index in range(len(self.bond_map)):
            if self.type[c1_index][0] == 'C':
                h_index = []
                c2_index = []
                for j in range(len(self.bond_map[c1_index])):
                    if self.bond_map[c1_index][j] == 1:
                        if self.type[j][0] == 'H':
                            h_index.append(j)
                        elif self.type[c1_index][0] == 'C':
                            c2_index.append(j)
                            
                if len(h_index) == 3:
                    # H \
                    # H - c1_index - c2_index - c3_index
                    # H /
                    for k in range(len(self.bond_map[c2_index[0]])):
                        if self.type[k][0] == 'C' and self.bond_map[c2_index[0]][k] == 1 and k != c1_index:
                            c3_index = k
                    v21 = self.coord[c1_index] - self.coord[c2_index[0]]
                    v12 = -v21
                    v23 = self.coord[c3_index] - self.coord[c2_index[0]]
                    upvector = np.cross(v21,v23)
                    upvector /= np.linalg.norm(upvector)
                    vh1 = np.matmul(R.from_rotvec(angle_CCH * upvector).as_dcm(),v12.transpose()).transpose()
                    vh1 *= length_CH/np.linalg.norm(vh1)
                    vh2 = np.matmul(R.from_rotvec(angle_HCH * v21/np.linalg.norm(v21)).as_dcm(),vh1.transpose()).transpose()
                    vh2 *= length_CH/np.linalg.norm(vh2)
                    vh3 = np.matmul(R.from_rotvec(2*angle_HCH * v21/np.linalg.norm(v21)).as_dcm(),vh1.transpose()).transpose()
                    vh3 *= length_CH/np.linalg.norm(vh3)
                    self.coord[h_index[0]] = np.array(vh1+self.coord[c1_index])
                    self.coord[h_index[1]] = np.array(vh2+self.coord[c1_index])
                    self.coord[h_index[2]] = np.array(vh3+self.coord[c1_index])
                elif len(h_index) == 2:
                    # H \        / c2_index_a
                    #    c1_index
                    # H /        \ c2_index_b
                    v2a1 = self.coord[c1_index] - self.coord[c2_index[0]]
                    v2b1 = self.coord[c1_index] - self.coord[c2_index[1]]
                    vmid = (v2a1/np.linalg.norm(v2a1)+v2b1/np.linalg.norm(v2b1))
                    vmid /= np.linalg.norm(vmid)
                    vh1 = np.matmul(R.from_rotvec(np.pi/2 * vmid).as_dcm(),v2a1.transpose()).transpose()
                    vh1 *= length_CH/np.linalg.norm(vh1)
                    vh2 = np.matmul(R.from_rotvec(np.pi/2 * vmid).as_dcm(),v2b1.transpose()).transpose()
                    vh2 *= length_CH/np.linalg.norm(vh2)
                    self.coord[h_index[0]] = np.array(vh1+self.coord[c1_index])
                    self.coord[h_index[1]] = np.array(vh2+self.coord[c1_index])
                    
                    

                    
        
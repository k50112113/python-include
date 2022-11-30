import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.optimize import leastsq
from ScatteringLengthTable import scattering_length_table
from AtomicFormFactorTable import atomic_form_factor_table
import ReadMD as RM

def smooth(x,y,n_points=1000,sg_filter=True,sg_window=101):
    xx = np.linspace(x.min(),x.max(),n_points)
    cs = CubicSpline(x, y)
    if sg_filter:
        sg = savgol_filter(cs(xx), sg_window, 3)
        return xx,sg
    else:
        return xx,cs(xx)
        
def sg_smooth(x,y,sg_window=11):
    return x,savgol_filter(y, sg_window, 3)

def sk_abs_prefactor(atom_weighting_name_,atom_num_,volume,weighting_type="neutron",k_values_=[]):
    #scattering length [fm]
    #volume [angstrom^3]
    #return [cm^-1]
    N = sum(atom_num_)
    if weighting_type == "neutron":
        sum_scattering_length_square = 0.0
        for i_particle in range(len(atom_weighting_name_)):
            sum_scattering_length_square += scattering_length_table[atom_weighting_name_[i_particle]]*atom_num_[i_particle]
        sum_scattering_length_square *= sum_scattering_length_square
        prefactor = sum_scattering_length_square/N/volume * 1e-26 * 1e24
        return prefactor
    elif weighting_type == 'xray':
        k_values_ = np.array(k_values_)
        atomic_form_factor_ = np.zeros((len(atom_num_),len(k_values_)))
        sum_atomic_form_factor_square = 0.0
        for i_particle in range(len(atom_weighting_name_)):
            for i_coefficient in range(0,len(atomic_form_factor_table[atom_weighting_name_[i_particle]])-1,2):
                atomic_form_factor_[i_particle] += atomic_form_factor_table[atom_weighting_name_[i_particle]][i_coefficient] * np.exp(-atomic_form_factor_table[atom_weighting_name_[i_particle]][i_coefficient+1]*((k_values_/4.0/np.pi)**2))
            atomic_form_factor_[i_particle] += atomic_form_factor_table[atom_weighting_name_[i_particle]][len(atomic_form_factor_table[atom_weighting_name_[i_particle]])-1]
            sum_atomic_form_factor_square += atomic_form_factor_[i_particle]*atom_num_[i_particle]
        sum_atomic_form_factor_square *= sum_atomic_form_factor_square
        sum_atomic_form_factor_square /= k_values_.shape[0]**2
        prefactor = sum_atomic_form_factor_square/N/volume * 1e-26 * 1e24
        return prefactor
    else:
        sum_atom_num_square = 0.0
        for i_particle in range(len(atom_weighting_name_)):
            sum_atom_num_square += atom_num_[i_particle]
        sum_atom_num_square *= sum_atom_num_square
        prefactor = sum_atom_num_square/N/volume * 1e-26 * 1e24
        return prefactor

def compute_absolute_cross_term_structure_factor(sk_n,sk_n1,sk_n2,pref_n,pref_n1,pref_n2):
    return sk_n*pref_n - sk_n1*pref_n1 - sk_n2*pref_n2 

def compute_normalized_cross_term_structure_factor(sk_n,sk_n1,sk_n2,pref_n,pref_n1,pref_n2):
    return 0.5*(compute_absolute_cross_term_structure_factor(sk_n,sk_n1,sk_n2,pref_n,pref_n1,pref_n2))/(pref_n1*pref_n2)**0.5

def make_mol_file(filename, output_filename, molecule_name = [], n_atom_in_molecule = [], n_type_in_molecule = []):
    data = RM.LAMMPS_ITR(filename)
    mol_atom_start_type_list_ = []
    type_index = 1
    for a_num_type in n_type_in_molecule:
        mol_atom_start_type_list_.append(type_index)
        type_index += a_num_type
    with open(output_filename,"w") as fout:
        fout.write("#\n%d\n"%(data.coord.shape[0]))
        molecule_id = 0
        i = 0
        while i < len(data.type):
            j = mol_atom_start_type_list_.index(data.type[i])
            molecule_id+=1
            for k in range(n_atom_in_molecule[j]):
                fout.write("%d %s\n"%(molecule_id,molecule_name[j]))
            i += n_atom_in_molecule[j]

def read(filename,quantity="",target_k=0,target_k_index=-1):
    
    with open(filename,"r") as fin:
        aline1 = fin.readline()
        
        if quantity == "":
            if "Self intermediate scattering function" in aline1:
                quantity="fskt"
            elif "Collective intermediate scattering function" in aline1:
                quantity="fkt"
            
        aline2 = fin.readline()
        
        if quantity == "":
            if "S(k)" in aline2:
                quantity = "sk"
            elif "g(r)" in aline2:
                quantity = "gr"
            elif "r2(t)" in aline2:
                if "Mutual" in aline1:
                    quantity = "mr2t"
                else:
                    quantity = "r2t"
            elif "alpha_2" in aline2:
                quantity = "alpha_2"
            elif "chi_4" in aline2:
                quantity = "chi_4"
            elif "C_jj" in aline2:
                quantity = "eacf"
        
        if quantity == "gr" \
        or quantity == "sk" \
        or quantity == "r2t" \
        or quantity == "mr2t" \
        or quantity == "alpha_2" \
        or quantity == "chi_4"\
        or quantity == "eacf":
            x_ = []
            y_ = []
            for aline in fin:
                if "#" not in aline:
                    linelist = aline.strip().split()
                    x_.append(float(linelist[0]))
                    y_.append(float(linelist[1]))
            
            return np.array(x_), np.array(y_), quantity
        elif quantity == "fskt" \
          or quantity == "fkt":
            k_ = []
            t_ = []
            f_ = []
            aline = fin.readline()
            while "#" not in aline:
                k_.append(float(aline.strip()))
                aline = fin.readline()
            for aline in fin:
                linelist = aline.strip().split()
                t_.append(float(linelist[0]))
                f_.append(np.array([float(i) for i in linelist[1:]]))
            
            k_ = np.array(k_)
            t_ = np.array(t_)
            f_tk = np.array(f_)
            f_kt = f_tk.transpose()
            if target_k > 0 and target_k_index > -1:
                print("Error: Both target_k and target_k_index specified.")
                exit()
            elif target_k > 0:
                f_kt_fitk = []
                gaussian = lambda p, x: p[0]*np.exp(-p[1]*(x**2))
                error_gaussian  = lambda p, x, y : y - gaussian(p, x)
                fit_coeff = []
                for a_f_tk in f_tk:
                    init  = [1, 0.001]
                    coeff = leastsq(error_gaussian, init, args=(k_, a_f_tk))[0]
                    f_kt_fitk.append(gaussian(coeff[:],target_k))
                return t_, np.array(f_kt_fitk), quantity                
            elif target_k_index > -1:
                return k_[target_k_index], t_, f_kt[target_k_index], quantity
            else:
                return k_, t_, f_kt, quantity


def write(quantity,input_filename,trajectory_file_path,output_file_path,\
          start_frame,end_frame,frame_interval,number_of_frames_to_average,\
          time_scale_type="log",trajectory_delta_time=1,time_interval=1.2,number_of_time_points=20,\
          calculation_type="atom",gro_file_path=None,molecule_file_path=None,dimension=3,\
          atom_name_1=None,atom_name_2=None,mass_1=None,mass_2=None,charge_1=None,charge_2=None,molecule_name_1=None,molecule_name_2=None,\
          weighting_type=None,atomic_form_factor_1=None,atomic_form_factor_2=None,scattering_length_1=None,scattering_length_2=None,\
          input_box_length=None,k_start_value=0,k_end_value=5,k_interval=0.01,\
          include_intramolecular=None,number_of_bins=400,max_cutoff_length=10,\
          overlap_length=1):
    quantity_function_ = {"sk" :"StructureFactor",\
                          "gr" :"PairDistributionFunction",\
                          "fskt":"SelfIntermediateScatteringFunction",\
                          "fkt":"CollectiveIntermediateScatteringFunction",\
                          "r2t":"MeanSquaredDisplacement",\
                          "mr2t":"MutualMeanSquaredDisplacement",\
                          "msd":"MeanSquaredDsiplacement",\
                          "chi4":"FourPointCorrelationFunction",\
                          "eacf":"ElectricCurrentAutocorrelationFunction"}
    quantity_function = quantity_function_[quantity]

    with open(input_filename,"w",newline='\n') as fout:
        fout.write('''-function=%s
-calculation_type=%s
-trajectory_file_path=%s'''%(quantity_function,calculation_type,trajectory_file_path))
        if gro_file_path:
            fout.write('''
-gro_file_path=%s'''%(gro_file_path))
        if molecule_file_path:
            fout.write('''
-molecule_file_path=%s'''%(molecule_file_path))
        fout.write('''
-output_file_path=%s'''%(output_file_path))
        fout.write('''
-start_frame=%s
-end_frame=%s
-frame_interval=%s
-dimension=%s
-number_of_frames_to_average=%s'''%(start_frame,end_frame,frame_interval,dimension,number_of_frames_to_average))
        if quantity in ['sk','fskt','fkt'] and weighting_type:
            fout.write('''
-weighting_type=%s'''%(weighting_type))
        ################################# Time correlation
        if quantity in ['msd','r2t','mr2t','fskt','fkt','chi4','eacf']:
            fout.write('''
-time_scale_type=%s
-trajectory_delta_time=%s
-time_interval=%s
-number_of_time_points=%s
#-time_array_indices='''%(time_scale_type,trajectory_delta_time,time_interval,number_of_time_points))
        ################################# Time correlation
        ################################# First atom group
        if atom_name_1:
            fout.write('''
-atom_name_1=%s'''%(atom_name_1))
        if molecule_name_1:
            fout.write('''
-molecule_name_1=%s'''%(molecule_name_1))
        if quantity in ['sk','fskt','fkt']:
            if atomic_form_factor_1:
                fout.write('''
-atomic_form_factor_1=%s'''%(atomic_form_factor_1))
            if scattering_length_1:
                fout.write('''
-scattering_length_1=%s'''%(scattering_length_1))
        if mass_1:
            fout.write('''
-mass_1=%s'''%(mass_1))
        if charge_1:
            fout.write('''
-charge_1=%s'''%(charge_1))
        ################################# First atom group
        ################################# Second atom group
        if atom_name_2:
            fout.write('''
-atom_name_2=%s'''%(atom_name_2))
        if molecule_name_2:
            fout.write('''
-molecule_name_2=%s'''%(molecule_name_2))
        if quantity in ['sk','fskt','fkt']:
            if atomic_form_factor_2:
                fout.write('''
-atomic_form_factor_2=%s'''%(atomic_form_factor_2))
            if scattering_length_2:
                fout.write('''
-scattering_length_2=%s'''%(scattering_length_2))
        if mass_2:
            fout.write('''
-mass_2=%s'''%(mass_2))
        if charge_2:
            fout.write('''
-charge_2=%s'''%(charge_2))
        ################################# Second atom group
        if quantity in ['sk','fskt','fkt']:
            fout.write('''
-k_start_value=%s
-k_end_value=%s
-k_interval=%s
#-max_k_number='''%(k_start_value,k_end_value,k_interval))
        if input_box_length:
            fout.write('''
-input_box_length=%s'''%(input_box_length))
        if quantity in ['gr']:
            if include_intramolecular == None:
                if molecule_file_path:
                    include_intramolecular = True
                else:
                    include_intramolecular = False
            fout.write('''
-include_intramolecular=%s'''%(include_intramolecular))
            fout.write('''
-number_of_bins=%s'''%(number_of_bins))
            fout.write('''
-max_cutoff_length=%s'''%(max_cutoff_length))
        if quantity in ['chi4']:
            fout.write('''
-overlap_length=%s'''%(overlap_length))




    
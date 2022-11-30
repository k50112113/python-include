import torch
import torch.nn as nn
from BPDescriptor import BPDescriptor, torchpi
from NeuralNetworkForceField import NeuralNetworkForceField, AtomicSubNetwork

class EnsembleNeuralNetworkForceFieldModel(NeuralNetworkForceFieldModel):
    def __init__(self, localtype_to_globaltype_list = None, localtype_indices = None, number_of_globaltypes = None, \
                       hidden_layer_size = None, atomic_subnetwork = [], \
                       Rc = 6.0, Rs_radial_step = 0.5, Rs_angular_step = 1.5, As_angular_step = 30, \
                       device = None, ensemble_size = 1):
        self.ensemble_size = ensemble_size
        super(EnsembleNeuralNetworkForceFieldModel, self).__init__(localtype_to_globaltype_list, localtype_indices, number_of_globaltypes, \
                                                                   hidden_layer_size, atomic_subnetwork, \
                                                                   Rc, Rs_radial_step, Rs_angular_step, As_angular_step, \
                                                                   device)
        
    def create_atomic_subnetwork(self, hidden_layer_size, atomic_subnetwork):
        self.input_size = self.bp_descriptor.get_descriptor_dimensions()
        self.ensemble_atomic_subnetwork_ = []
        print("\t\tinput dimension = %d"%(self.input_size))
        for i_ensemble in range(self.ensemble_size):
            self.ensemble_atomic_subnetwork_.append([])
            for i_localtype in range(self.number_of_localtypes_):
                i_globaltype = self.localtype_to_globaltype_list_[i_localtype]
                if atomic_subnetwork[i_localtype] is None:
                    self.ensemble_atomic_subnetwork_[i_ensemble].append(AtomicSubNetwork(self.input_size, hidden_layer_size[i_globaltype], i_globaltype).to(self.device))
                else:
                    if atomic_subnetwork[i_ensemble][i_localtype].input_size == self.input_size:
                        self.ensemble_atomic_subnetwork_[i_ensemble].append(atomic_subnetwork[i_localtype])
                    else:
                        print("Error: an imported Atomic Sub-network has incompatible descriptor size between the BPDescriptor and AtomicSubNetwork.")

    def evaluate_energy(self, atom_position, lattice_vector, inverse_lattice_vector):
        # atom_position: (N*Dim)
        descriptor = self.bp_descriptor.compute_descriptor(atom_position, lattice_vector, inverse_lattice_vector)
        
        ensemble_total_energy = torch.zeros(self.ensemble_size).to(self.device)
        for i_ensemble in range(self.ensemble_size):
            for j_type in range(self.number_of_localtypes_):
                atomic_energy = self.ensemble_atomic_subnetwork_[i_ensemble][j_type].forward(descriptor[j_type])
                ensemble_total_energy[i_ensemble] += atomic_energy.squeeze(-1).sum(dim=-1)
        return ensemble_total_energy
    
    def evaluate_force(self, atom_position, lattice_vector, inverse_lattice_vector):
        atom_position.requires_grad = True
        if atom_position.grad: atom_position.grad.zero_()
        ensemble_total_energy = self.evaluate_energy(atom_position,lattice_vector,inverse_lattice_vector)
        ensemble_total_force = torch.zeros((self.ensemble_size,*atom_position.shape)).to(self.device)
        for i_ensemble in range(self.ensemble_size):
            ensemble_total_force[i_ensemble] = -torch.autograd.grad(ensemble_total_energy[i_ensemble], atom_position, create_graph=True)[0]
        return ensemble_total_force, ensemble_total_energy

    def evaluate_virial(self, atom_position, lattice_vector, inverse_lattice_vector):
        if self.unit == 0:
            prefactor = 4184/6.02214076e-7/101325 # kcal/mole/A^3 -> atm
        else:
            prefactor = 1.602176565e+6 # eV/A^3 -> bar
        inverse_volume = 1.0/torch.linalg.det(lattice_vector)
        prefactor*=inverse_volume
        lattice_vector.requires_grad = True
        if lattice_vector.grad: lattice_vector.grad.zero_()
        # inv_lattice_vector = lattice_vector.inverse()
        ensemble_total_force, ensemble_total_energy = self.evaluate_force(atom_position, lattice_vector, inverse_lattice_vector)
        ensemble_total_virial = torch.zeros((self.ensemble_size,3,3)).to(self.device)
        ensemble_total_pressure =  torch.zeros(self.ensemble_size).to(self.device)
        for i_ensemble in range(self.ensemble_size):
            lattice_virial = torch.matmul(torch.autograd.grad(total_energy, lattice_vector)[0].transpose(-2,-1), lattice_vector)*prefactor
            force_virial = -torch.matmul(total_force.transpose(-2,-1), atom_position)*prefactor
            total_virial = -(force_virial+lattice_virial)
            total_virial = (total_virial.transpose(0,1) + total_virial)*0.5
            total_pressure = total_virial.diag().mean()
            ensemble_total_virial[i_ensemble] = total_virial
            ensemble_total_pressure[i_ensemble] = total_pressure

        return ensemble_total_pressure, ensemble_total_virial, ensemble_total_force, ensemble_total_energy
    
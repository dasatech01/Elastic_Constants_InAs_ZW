import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import h5py
import mp_api.client
from mp_api.client import MPRester

# This script retrieves the elastic constants for InAs in the Zinc Blende (ZW) structure from the Materials Project database.
with MPRester("2zenyrAKUFGB1dF4Qg9aJMiBAeWatnpY") as mpr:
    elasticity_InAs_ZW = mpr.materials.elasticity._search(material_ids=["mp-20305"], fields=["elastic_tensor"])


# The retrieved data is then processed to extract the elastic tensor and convert it into a matrix format
elasticity_InAs_ZW_matrix = []

for doc in elasticity_InAs_ZW:
    elasticity_InAs_ZW_matrix.append(doc.elastic_tensor.raw)

elasticity_InAs_ZW_matrix = np.array(elasticity_InAs_ZW_matrix)


# This script extracts the elastic constants from an HDF5 file, which contains pre-computed elastic constants for InAs in the 
# Zinc Blende structure at various mesh cutoffs.
filename = '/Users/ahnafsaminakter/Documents/UofT/Research/Prof Ruda/Files/Scripts/InAs_ZW_TE_EC_iteration.hdf5'

f = h5py.File(filename, 'r')

# print("Keys in the file:")
# for key in f.keys():
#     print(key)
#     print(f"Type of {key}: {type(f[key])}")

# for i in range(len(list(f.keys()))):
#     print(f"Group {i}: {list(f.keys())[i]}")

Elastic_Constants = []

for i in range(10, 155, 5):
    Elastic_Constants.append(list(f[f"elastic_constants_ElasticConstants_mesh_cutoff_{i}.0 Hartree"]['elastic_constants']['array']['data']))


# We now check the convergence of the elastic constants with respect to the mesh cutoff.
dM_norm = []
for i in range(len(Elastic_Constants)):
    dM_norm.append(LA.norm((elasticity_InAs_ZW_matrix) - np.array(Elastic_Constants[i])))

plt.plot(range(10, 155, 5), dM_norm, marker='o')
plt.xlabel('Mesh Cutoff (Hartree)')
plt.ylabel('Norm of Difference')
plt.title('Convergence of Elastic Constants for InAs in ZW Structure')
plt.show()

# Now we do it for formation energy.

formation_energy = []

for i in range(10, 155, 5):
    formation_energy.append(np.array(f[f"total_energy_TotalEnergy_mesh_cutoff_{i}.0 Hartree"]['energy_components']['Kinetic']['array']['data']))

plt.plot(range(10, 155, 5), formation_energy, marker='o')
plt.xlabel('Mesh Cutoff (Hartree)')
plt.ylabel('Kinetic Energy (Hartree)')
plt.title('Convergence of Kinetic Energy for InAs in ZW Structure')
plt.show()
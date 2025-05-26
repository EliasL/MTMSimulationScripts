from contiPotential import ContiEnergy
from matplotlib import pyplot as plt
import numpy as np

strain = np.linspace(0.0, 1, 100)
strain1 = np.linspace(0.5, 1, 100)
e = ContiEnergy.energy_from_simpleShear(strain)
e1 = ContiEnergy.energy_from_simpleShear(strain1)
plt.plot(strain, e)
# plt.show()

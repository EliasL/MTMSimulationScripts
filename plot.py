from MTMath.plotEnergy import (
    generate_energy_grid,
    OneDPotential,
    make3DEnergyField,
    plotEnergyField,
)

g, x, y = generate_energy_grid(resolution=1000, return_XY=True)
# plotEnergyField(g)
# OneDPotential()
make3DEnergyField(g, x, y, zoom=0.2, energy_lim=[None, 4.15])

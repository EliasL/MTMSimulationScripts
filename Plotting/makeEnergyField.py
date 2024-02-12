from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os


from settings import settings

def makeEnergyField(path, csv_file):
    print("Plotting energy field...")

    dataPath = path + settings["DATAFOLDERPATH"]
    filePath = dataPath + csv_file

    if not os.path.exists(filePath):
        print(f"No file found at: {filePath}")
        return

    # Reading data from CSV
    data = np.genfromtxt(filePath, delimiter=',')
    
    # Replace NaN and -NaN with infinity
    data[np.isnan(data)] = np.inf

    # Extracting x, y, and energy values
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    energies = data[:, 2]

    # Assuming equal spacing and regular grid
    grid_size = int(np.sqrt(len(energies)))
    energy_grid = energies.reshape((grid_size, grid_size)).transpose()


    # Create the plot
    plt.figure(figsize=(8, 6))

    # Set the minimum and maximum values for the color bar
    min_energy = energy_grid.min()  # Replace with your desired minimum value
    max_energy = 4.16  # Replace with your desired maximum value

    plt.imshow(energy_grid, cmap='viridis', origin='lower', vmin=min_energy, vmax=max_energy)

    # Add a thin black circle
    circleSize = grid_size/2
    circle_center_x = circleSize
    circle_center_y = circleSize
    circle = Circle((circle_center_x, circle_center_y), circleSize, color='black', fill=False, linewidth=1)
    plt.gca().add_patch(circle)

    # Adjusting ticks
    plt.xticks(np.linspace(0, grid_size - 1, 5), np.linspace(x_vals.min(), x_vals.max(), 5).round(2))
    plt.yticks(np.linspace(0, grid_size - 1, 5), np.linspace(y_vals.min(), y_vals.max(), 5).round(2))

    plt.colorbar(label='Energy field in a Poncare disk')
    nbs = u'\u00A0'  #non-breaking-space
    plt.xlabel('T(Length ratio)')
    plt.ylabel(f'← Large angle {nbs*7} T(Length ratio and θ - π/2) {nbs*7} Small angle →')
    plt.title('Energy')

    output_pdf_path = path + "energy_field.pdf"
    plt.savefig(output_pdf_path, format='pdf')


if __name__ == "__main__":
    # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    makeEnergyField('build/output/testing/','energy_grid.csv')
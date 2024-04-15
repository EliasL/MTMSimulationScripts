from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from settings import settings

def C2PoincareDisk(C):

    if C.ndim == 2:
        x_, y_ = (C[0, 1]/C[1, 1], np.sqrt(np.linalg.det(C))/C[1, 1])
    else:
        dets = np.linalg.det(C)
        x_, y_ = (C[:, 0, 1]/C[:, 1, 1], np.sqrt(dets)/C[:, 1, 1])

    x = (x_**2+y_**2-1)/(x_**2+(y_+1)**2)
    y = 2*x_/(x_**2+(y_+1)**2)

    return x, y

def drawC(ax, C, scale):
    pos = C2PoincareDisk(C)
    ax.plot(pos[0]*scale/2 + scale/2, pos[1]*scale/2+scale/2, 
            c='black', linewidth=0.6,
            linestyle='--')

def drawFundamentalDomain(ax, scale):
    nr = 1000
    one = np.array([1]*nr)
    zero = np.array([0]*nr)
    # VERTICAL LINE
    t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2/np.sqrt(3)), nr))
    # Values from -1<t<1 give complex solutions
    # det=1, C12=C21, C11=C22
    C = np.array([[t, np.sqrt(t**2-1)],
                    [np.sqrt(t**2-1), t]]).transpose(2,0,1)

    drawC(ax, C, scale)
    

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    C = np.array([[t, zero],
                    [zero, 1/t]]).transpose(2,0,1)
    drawC(ax, C, scale)
    
    
    # FUNDAMENTAL DOMAIN (0.01 to avoid div by 0)
    # https://www.wolframalpha.com/input?i=0%3Ca%3Cd%2C+b%3Da%2F2%2C+++a*d-b*c%3D1%2C+b%3Dc
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(2/np.sqrt(3)), nr))
    # Negative values are outside of the circle
    # det=1, C12=C21,   
    C = np.array([[t,    t/2],
                    [t/2, (t**2+4)/(4*t)]]).transpose(2,0,1)
    drawC(ax, C, scale)

def makeEnergyField(csv_file):
    print("Plotting energy field...")

    if not os.path.exists(csv_file):
        print(f"No file found at: {csv_file}")
        return

    # Reading data from CSV
    data = np.genfromtxt(csv_file, delimiter=',')
    
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
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()

    # Set the minimum and maximum values for the color bar
    min_energy = energy_grid.min()  # Replace with your desired minimum value
    max_energy = 4.16  # Replace with your desired maximum value

    img = ax.imshow(energy_grid, cmap='viridis', origin='lower', vmin=min_energy, vmax=max_energy)

    # Add a thin black circle
    circleSize = grid_size/2
    circle_center_x = circleSize
    circle_center_y = circleSize
    circle = Circle((circle_center_x, circle_center_y), circleSize, color='black', fill=False, linewidth=1)
    fig.gca().add_patch(circle)

    # Draw fundamental domain
    drawFundamentalDomain(ax, grid_size)

    # Adjusting ticks
    ax.set_xticks(np.linspace(0, grid_size - 1, 5), np.linspace(x_vals.min(), x_vals.max(), 5).round(2))
    ax.set_yticks(np.linspace(0, grid_size - 1, 5), np.linspace(y_vals.min(), y_vals.max(), 5).round(2))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Add colorbar
    cbar = fig.colorbar(img, label='Energy')
    default_font_size = plt.rcParams['font.size']  # Fetch default font size
    cbar.ax.set_title(f'Capped at ${max_energy}$', fontsize=default_font_size)
    nbs = u'\u00A0'  #non-breaking-space
    ax.set_xlabel(f'← Tall {nbs*7} $P_x$(Length ratio) {nbs*7} Wide →')
    ax.set_ylabel(f'← Large angle {nbs*7} $P_y$(Length ratio and $\\theta - \\pi/2$) {nbs*7} Small angle →')
    ax.set_title('Energy field in a Poincaré disk')

    path = os.path.dirname(csv_file)
    output_pdf_path = os.path.join(path, "energy_field.pdf")
    fig.savefig(output_pdf_path, format='pdf', dpi=600,
                bbox_inches='tight', pad_inches=0)


def make3DEnergyField(csv_file):
    print("Plotting energy field...")

    if not os.path.exists(csv_file):
        print(f"No file found at: {csv_file}")
        return

    # Reading data from CSV
    data = np.genfromtxt(csv_file, delimiter=',')
    
    # Replace NaN and -NaN with infinity
    data[np.isnan(data)] =np.inf

    # Extracting x, y, and energy values
    x_vals = data[:, 0]
    y_vals = data[:, 1]
    energies = data[:, 2]

    # Assuming equal spacing and regular grid
    grid_size = int(np.sqrt(len(energies)))
    energy_grid = energies.reshape((grid_size, grid_size)).transpose()

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Creating the x, y meshgrid (assumes equal spacing)
    X, Y = np.meshgrid(np.linspace(x_vals.min(), x_vals.max(), grid_size),
                       np.linspace(y_vals.min(), y_vals.max(), grid_size))

    # Set the minimum and maximum values for the Z axis
    min_energy = energy_grid.min()
    max_energy = 4.16
    energies[energies>max_energy]=max_energy
    # Calculate the radius for each point on the grid
    radii = np.sqrt(X**2 + Y**2)

    # Create a mask for points outside the circle of radius 0.8
    mask = radii > 0.7

    # Apply the mask to the energy grid to set values outside the radius to np.nan
    energy_grid_masked = np.copy(energy_grid)
    energy_grid_masked[mask] = np.nan

    # Plot the surface with the masked energy grid
    surf = ax.plot_surface(X, Y, energy_grid_masked, cmap='viridis', linewidth=0, antialiased=False, vmin=min_energy, vmax=max_energy)

    # Add a color bar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Energy field in a Poincaré disk')
    cbar.ax.text(1.4, max_energy, f'Capped at {max_energy}', va='center')  # text annotation

    # Set labels
    ax.set_xlabel('$P$(Length ratio)')
    ax.set_ylabel(f'← Large angle $P$(Length ratio and $θ - π/2$) Small angle →')
    ax.set_zlabel('Energy')
    ax.set_title('Energy Surface Plot')

    # Adjust limits and view angle if necessary
    ax.set_zlim(min_energy, max_energy)
    ax.view_init(elev=30, azim=30)  # elevation and angle

    # Save the figure
    output_pdf_path = os.path.join(os.path.dirname(csv_file), "energy_field_3D.pdf")
    plt.savefig(output_pdf_path, format='pdf')
    plt.show()

if __name__ == "__main__":
    # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    makeEnergyField('/Users/eliaslundheim/work/PhD/MTS2D/build/energy_grid.csv')
    #make3DEnergyField('/Users/eliaslundheim/work/PhD/MTS2D/build/energy_grid.csv')
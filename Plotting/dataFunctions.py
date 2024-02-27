import math
import numpy as np
import xml.etree.ElementTree as ET
import vtk
from vtkmodules.vtkCommonDataModel import vtkPolyData

# Set the log level to ERROR to only see error messages
# This is to avoid seeing:  2024-02-26 15:09:13.742 (  17.205s) [          80BB7A]    vtkExtractEdges.cxx:427   INFO| Executing edge extractor: points are renumbered
#                           2024-02-26 15:09:13.743 (  17.205s) [          80BB7A]    vtkExtractEdges.cxx:543   INFO| Created 261 edges
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)

from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm

def parse_pvd_file(path, pvd_file):
    tree = ET.parse(path+pvd_file)
    root = tree.getroot()
    vtu_files = []

    for dataset in root.iter('DataSet'):
        vtu_files.append(dataset.attrib['file'])

    return vtu_files

def convert_vtu_to_polydata(vtu_file_path) -> vtkPolyData:
    # Read the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()

    # Use vtkGeometryFilter to convert the unstructured grid to polydata
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputConnection(reader.GetOutputPort())
    geometryFilter.Update()

    # The output of the geometry filter is vtkPolyData
    polydata = geometryFilter.GetOutput()

    return polydata

def getDataFromName(nameOrPath):
    # Split the filename by underscores
    fileName = nameOrPath.split('/')[-1]
    parts = fileName.split('_')

    # Initialize an empty dictionary
    result = {}

    # We skipp the first and last part. The first part is the 'name', the last
    # part is the type, ie .N.vtu
    result['name'] = parts[0]
    for part in parts[1:-1]:
        key, value = part.split('=')
        # Add the key-value pair to the dictionary
        result[key] = value

    return result

def getDataSize(path, vtu_files):
    """
    Calculate the size of the data from VTU files.

    Args:
        path (str): The path to the directory containing the VTU files.
        vtu_files (list of str): A list of VTU filenames.

    Returns:

        nrSteps, 
        nrNodes,
        nrElements
    """

    # To find the number of nodes and elements, we need to open a file
   
    nodes, stress_field, energy_field = read_vtu_data(vtu_files[0]) 

    # Number of nodes
    nrNodes = len(nodes)

    # Number of elements
    nrElements = len(energy_field)

    return nrNodes, nrElements




class Visualizer:
    def __init__(self, framePath, vtu_files, width=1920, height=1080,
                 show_nodes=True, show_text=True, show_wireframe=True, show_stress=True):
        self.framePath = framePath
        self.vtu_files = vtu_files
        self.show_nodes = show_nodes
        self.show_text = show_text
        self.show_wireframe = show_wireframe
        self.show_stress = show_stress

        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.polydata = convert_vtu_to_polydata(vtu_files[-1]) # Use the final position to init the images
        
        self.renderWindowInteractor.Initialize()
        self.renderWindow.SetSize(width, height)
        self.setup_actors()
        self.configure_camera()

        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark grey color

    def setup_actors(self):
        if self.show_text:
            self.add_text_actor()
        if self.show_nodes:
            self.add_nodes_actor()
        if self.show_wireframe:
            self.add_wireframe_actor()
        if self.show_stress:
            self.add_stress_actor()

    def add_text_actor(self):
        # Create a text actor
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("Initial Text")
        text_property = self.text_actor.GetTextProperty()
        text_property.SetFontSize(24)
        text_property.SetColor(1.0, 1.0, 1.0)  # White color
        
        # Get the size of the render window to position the text actor
        width, height = self.renderWindow.GetSize()
        
        # Position the text actor at the top-left corner of the window
        self.text_actor.SetPosition(10, height - 100)
        
        # Add the text actor to the renderer
        self.renderer.AddActor(self.text_actor)

    def add_wireframe_actor(self):
        # Create a wireframe representation
        self.wireframe = vtk.vtkExtractEdges()
        self.wireframe.SetInputData(self.polydata)

        wireframe_mapper = vtk.vtkPolyDataMapper()
        wireframe_mapper.SetInputConnection(self.wireframe.GetOutputPort())

        wireframe_actor = vtk.vtkActor()
        wireframe_actor.SetMapper(wireframe_mapper)
        wireframe_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # Set wireframe color, e.g., white
        wireframe_actor.GetProperty().SetLineWidth(1)  # Set wireframe line width

        self.renderer.AddActor(wireframe_actor)

    def add_nodes_actor(self):
        # Create a source for the glyphs (e.g., spheres)
        glyph_source = vtk.vtkSphereSource()
        glyph_source.SetRadius(0.2)  # Adjust the radius as needed

        # Create the glyphs
        self.glyph = vtk.vtkGlyph3D()
        self.glyph.SetSourceConnection(glyph_source.GetOutputPort())
        self.glyph.SetInputData(self.polydata)
        self.glyph.ScalingOff()  # Turn off scaling if your points don't have varying sizes
        
        self.glyph.Update()

        # Use the output of the glyph filter for the mapper
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(self.glyph.GetOutputPort())

        # Create an actor for the glyphs
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)

        # Add the glyph actor to the renderer
        self.renderer.AddActor(glyph_actor)

        self.polydata.Modified()
        glyph_actor.GetMapper().Update()



    def find_global_stress_range(self, stress_array_name):
        min_magnitude = float('inf')
        max_magnitude = -float('inf')
        for vtu_file in self.vtu_files:
            polydata = convert_vtu_to_polydata(vtu_file)
            # Check both cell and point data for the stress array
            if polydata.GetPointData().HasArray(stress_array_name):
                stress_field = polydata.GetPointData().GetArray(stress_array_name)
            else:
                raise(Exception("Data not found"))
            # Calculate magnitude for each vector and update global min/max
            num_points = polydata.GetNumberOfPoints()
            for i in range(num_points):
                stress_vector = stress_field.GetTuple3(i)
                magnitude = np.linalg.norm(stress_vector)
                min_magnitude = min(min_magnitude, magnitude)
                max_magnitude = max(max_magnitude, magnitude)
        return (min_magnitude, max_magnitude)

    def add_stress_magnitude_to_polydata(self):
        # Assuming 'self.polydata' is your mesh and it already contains 'stress_field'
        stress_field = self.polydata.GetPointData().GetArray("stress_field")

        # Create a new array for stress magnitudes
        num_points = self.polydata.GetNumberOfPoints()

        stress_magnitude = vtk.vtkFloatArray()
        stress_magnitude.SetNumberOfValues(num_points)
        stress_magnitude.SetName("stress_magnitude")

        # Calculate the magnitude of stress for each point and store it
        for i in range(num_points):
            stress_vector = stress_field.GetTuple3(i)
            magnitude = np.linalg.norm(stress_vector)
            stress_magnitude.SetValue(i, magnitude)

        # Add the magnitude array to the point data
        self.polydata.GetPointData().AddArray(stress_magnitude)
        self.polydata.GetPointData().SetActiveScalars("stress_magnitude")

    def make_colors_and_color_bar(self):
        stress_range = self.find_global_stress_range("stress_field")
        # Create a lookup table (LUT) for mapping stress values to colors
        print(stress_range)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)

        # Create a color transfer function that will be used for the LUT
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(stress_range[0], 0.0, 0.0, 1.0)  # Blue at the minimum stress
        colorTransferFunction.AddRGBPoint((stress_range[1] - stress_range[0]) / 2.0 + stress_range[0], 1.0, 1.0, 1.0)  # White at the midpoint
        colorTransferFunction.AddRGBPoint(stress_range[1], 1.0, 0.0, 0.0)  # Red at the maximum stress

        # Apply the color transfer function to the lookup table
        for i in range(256):
            rgb = colorTransferFunction.GetColor(float(i)/255.0 * (stress_range[1] - stress_range[0]) + stress_range[0])
            lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
        lut.SetRange(stress_range)  # Set this to the actual stress range
        lut.SetTableRange(stress_range)  # Set this to the actual stress range


        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetTitle("Stress Magnitude")
        scalarBar.SetDrawAnnotations(False)
        scalarBar.SetNumberOfLabels(3)
        textProperty = vtk.vtkTextProperty()
        textProperty.SetFontSize(30)
        scalarBar.SetAnnotationTextProperty(textProperty)
        scalarBar.SetTitleTextProperty(textProperty)
        scalarBar.SetLabelTextProperty(textProperty)
        scalarBar.AnnotationTextScalingOff()
        scalarBar.SetUnconstrainedFontSize(True)

        # Adjust the size of the color bar
        scalarBar.SetWidth(0.05)  # Adjust the width as a proportion of the viewport size
        scalarBar.SetHeight(0.3)  # Adjust the height as a proportion of the viewport size
        # Add the scalar bar to the renderer
        scalarBar.SetLookupTable(lut)
        scalarBar.SetCustomLabels(["1", '2', '3'])
        scalarBar.Modified()
        self.renderer.AddActor(scalarBar)
        return lut


    def add_stress_actor(self):
        lut = self.make_colors_and_color_bar()

        self.add_stress_magnitude_to_polydata()

        # Visualization setup remains the same as before
        self.stress_mapper = vtk.vtkPolyDataMapper()
        self.stress_mapper.SetInputData(self.polydata)
        self.stress_mapper.SetLookupTable(lut)
            # self.stress_mapper.SetScalarRange(stress_range)
        self.stress_mapper.SetScalarModeToUsePointData()
        self.stress_mapper.SelectColorArray("stress_magnitude")

        self.stress_actor = vtk.vtkActor()
        self.stress_actor.SetMapper(self.stress_mapper)
        self.renderer.AddActor(self.stress_actor)



    def configure_camera(self):
        bounds = self.polydata.GetPoints().GetBounds()
        
        # Calculate the center of the bounds
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2

        # Calculate the largest dimension of the bounding box
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]

        camera = self.renderer.GetActiveCamera()
        # Calculate the field of view in radians
        fov = camera.GetViewAngle()
        fov_radians = math.radians(fov)

        render_width, render_height = self.renderWindow.GetSize()
        aspect_ratio = render_width / render_height
        # Adjust the field of view for aspect ratio
        if width / height > aspect_ratio:
            # Use horizontal FOV for distance calculation
            hfov_radians = 2 * math.atan(math.tan(fov_radians / 2) * aspect_ratio)
            distance = (width / 2.0) / math.tan(hfov_radians / 2.0)
        else:
            # Use vertical FOV for distance calculation
            distance = (height / 2.0) / math.tan(fov_radians / 2.0)

        # Set camera position and focal point
        camera.SetFocalPoint(center_x, center_y, center_z)
        camera.SetPosition(center_x, center_y, center_z + distance+2)

    def update_text_actor(self, frame_index, energy_field, metadata):
        average_energy = sum(energy_field) / len(energy_field) if energy_field is not None else 0
        lines = [
            f"State: {metadata['name']}",
            f"Frame: {frame_index}",
            f"Load: {metadata['load']}",
            f"Average Energy: {average_energy:.2f}"
        ]
        self.text_actor.SetInput("\n".join(lines))

    def create_frames(self):
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.renderWindow)
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())

        for frame_index, vtu_file in enumerate(tqdm(self.vtu_files)):

            self.polydata = convert_vtu_to_polydata(vtu_file)
            if self.show_nodes:
                self.glyph.SetInputData(self.polydata)
            if self.show_wireframe:
                self.wireframe.SetInputData(self.polydata)
            if self.show_stress:
                self.add_stress_magnitude_to_polydata()
                self.stress_mapper.SetInputData(self.polydata)
            if self.show_text:
                metadata = getDataFromName(vtu_file)
                _, _, energy_field = read_vtu_data(vtu_file)
                self.update_text_actor(frame_index, energy_field, metadata)
            
            self.polydata.Modified()
            self.renderWindow.Render()
            # Update the window to image filter and write the current frame to a file
            window_to_image_filter.Modified()
            window_to_image_filter.Update()  # Ensure the filter processes the render window contents
            writer.SetFileName(f"{self.framePath}/frame_{frame_index:04d}.png")
            writer.Write()
import subprocess
import os

from settings import settings
from vtkFunctions import *

def select_vtu_files(vtu_files, nrSteps):
    # Always include the first and last frames
    if len(vtu_files) <= 2 or nrSteps <= 2:
        return vtu_files

    # Calculate the step size
    step_size = int(max(1, len(vtu_files) // (nrSteps - 1)))

    # Select files at regular intervals
    selected_files = vtu_files[::step_size]

    # Ensure the last file is included, if it's not already
    if selected_files[-1] != vtu_files[-1]:
        selected_files.append(vtu_files[-1])

    return selected_files

# Use ffmpeg to convert a folder of .png images into a mp4 file
def makeAnimations(path, pvd_file):
   
    print("Creating frames...")

    dataPath = path + settings["DATAFOLDERPATH"]   
    framePath = path + settings["FRAMEFOLDERPATH"]  
    if(not os.path.exists(path+pvd_file)):
        print(f"No file found at: {path+pvd_file}")
        return
    
    vtu_files = parse_pvd_file(path, pvd_file)

    # we don't want every frame to be created, so in order to find out what
    # frames should be drawn, we first check how much load change there is
    print(vtu_files[0])
    first = getDataFromName(vtu_files[0])    
    last = getDataFromName(vtu_files[-1])    
    loadChange = float(last['load']) - float(first['load'])

    # Length of video in seconds
    videoLength = 15 * loadChange
    # Define the frame rate
    fps = 30
    nrSteps = videoLength*fps

    # we select a reduced number of frames
    vtu_files = select_vtu_files(vtu_files, nrSteps)

    if len(vtu_files)<nrSteps:
        # If we don't have enough frames, we need to make each frame last longer
        # We will make the video last 7 seconds
        fps = len(vtu_files)/7

    nrNodes, nrElements = getDataSize(dataPath, vtu_files)
    makeImages(framePath, dataPath, vtu_files)
    
    print("Creating animations...")

    # Define the output video filename
    # The name of the video is the same as the name of the folder+_video.mp4
    outputVideo = path+path.split('/')[-2]+'_video.mp4'
    framesPath = path+settings["FRAMEFOLDERPATH"]

    # Construct the FFmpeg command as a string
    ffmpeg_command = (
        f"ffmpeg -y -r {fps} -pattern_type glob -i '{framesPath}/*.png' "
        f"-c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p {outputVideo}"
    )
    # Run the FFmpeg command using subprocess, and hide output
    result = subprocess.run(ffmpeg_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Check if the command was successful
    if result.returncode == 0:
        pass
    else:
        # If you see this, remove the stdout and stderr keywords from the subprocess
        # for more details on the error. (Have you installed ffmpeg?)
        print("ffmpeg command failed with return code", result.returncode)

if __name__ == "__main__":
    output = '/media/elias/dataStorage/MTS2D_output/smallSimulation/'

    # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    makeAnimations(output,'collection.pvd')
import os
import cv2
import imageio.v2 as imageio  # Adjusted import here


from settings import settings
from pyplotFunctions import makeImages, plot_mesh, plot_nodes
from dataFunctions import parse_pvd_file, getDataFromName

# This function selects a subset of the vtu files to speed up the animation
# process. (For example, if the video would be 2 hours long, or have a fps of
# 2000, there is no need to use all the frames, so we skip a few)
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

def framesToMp4(frames, outFile, fps): 
    print("Creating mp4...")
    # Determine the width and height from the first image
    image_path = frames[0]
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(outFile, fourcc, fps, (width, height))

    for image_path in frames:
        frame = cv2.imread(image_path)        
        # Assuming all images are the same size, add frame to video
        out.write(frame)

    # Release everything if job is finished
    out.release()
 
def framesToGif(frames, outfile, fps):
    print("Making Gif...")
    frames = []
    for image_path in frames:
        frame = imageio.imread(image_path)
        frames.append(frame)

    # Save the frames as a GIF
    imageio.mimsave(outfile, frames, 'GIF', duration = 1/fps, loop=0)

 
# Use ffmpeg to convert a folder of .png frames into a mp4 file
def makeAnimations(path, pvd_file):
   
    framePath = path + settings["FRAMEFOLDERPATH"]  
    if(not os.path.exists(path+pvd_file)):
        print(f"No file found at: {path+pvd_file}")
        return
    
    vtu_files = parse_pvd_file(path, pvd_file)

    # we don't want every frame to be created, so in order to find out what
    # frames should be drawn, we first check how much load change there is
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

    images = makeImages(plot_mesh, framePath, vtu_files)

    # Define the path and file name
    # The name of the video is the same as the name of the folder+_video.mp4
    outPath = path+path.split('/')[-2]+'_video.mp4'
    framesToMp4(images, outPath, fps)

if __name__ == "__main__":
    output = '/Volumes/data/MTS2D_output/s100x100l0.15,0.001,1t3s0/'

    # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    makeAnimations(output,'collection.pvd')
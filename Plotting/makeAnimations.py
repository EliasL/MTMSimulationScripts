import os
import subprocess
import cv2
import imageio.v2 as imageio  # Adjusted import here


from settings import settings
from pyplotFunctions import make_images, plot_mesh, plot_nodes
from dataFunctions import parse_pvd_file, get_data_from_name
from makePvd import create_collection


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
    print(f"Creating {outFile}")
    # Determine the width and height from the first image
    image_path = frames[0]
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(outFile, fourcc, fps, (width, height))

    for image_path in frames:
        frame = cv2.imread(image_path)
        # Assuming all images are the same size, add frame to video
        out.write(frame)

    # Release everything if job is finished
    out.release()


def framesToGif(frames, outFile, fps):
    print(f"Making {outFile} Gif...")
    frames = []
    for image_path in frames:
        frame = imageio.imread(image_path)
        frames.append(frame)

    # Save the frames as a GIF
    imageio.mimsave(outFile, frames, "GIF", duration=1 / fps, loop=0)


# Use ffmpeg to convert a folder of .png frames into a mp4 file
def makeAnimations(path, macro_data=None, pvd_file=None):
    framePath = path + settings["FRAMEFOLDERPATH"]
    if macro_data is None:
        macro_data = path + settings["MACRODATANAME"] + ".csv"
    if pvd_file is None:
        pvd_file = path + settings["COLLECTIONNAME"] + ".pvd"

    if not os.path.exists(pvd_file):
        print(f"No file found at: {pvd_file}")
        print("Creating pvd file...")
        create_collection(path + settings["DATAFOLDERPATH"])

    vtu_files = parse_pvd_file(path, pvd_file)

    # we don't want every frame to be created, so in order to find out what
    # frames should be drawn, we first check how much load change there is
    first = get_data_from_name(vtu_files[0])
    last = get_data_from_name(vtu_files[-1])
    loadChange = float(last["load"]) - float(first["load"])

    # Length of video in seconds
    videoLength = 15 * loadChange
    # Define the frame rate
    fps = 30
    nrSteps = videoLength * fps

    # we select a reduced number of frames
    vtu_files = select_vtu_files(vtu_files, nrSteps)

    if len(vtu_files) < nrSteps:
        # If we don't have enough frames, we need to make each frame last longer
        # We will make the video last 7 seconds
        fps = len(vtu_files) / 7

    # Define the path and file name
    # The name of the video is the same as the name of the folder+_video.mp4
    for function, fileName in zip([plot_mesh, plot_nodes], ["mesh", "nodes"]):
        images = make_images(function, framePath, vtu_files, macro_data)
        outPath = path + path.split("/")[-2] + f"_{fileName}_video.mp4"
        framesToMp4(images, outPath, fps)
        subprocess.run(
            [
                "gifski",
                outPath,
                "-o",
                f"{path + path.split('/')[-2]}_{fileName}_video.gif",
            ]
        )


if __name__ == "__main__":
    output = "/Volumes/data/KeepSafe/simpleShear,s150x150l0.15,2e-05,1PBCt4EpsG0.01s0/"

    # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    makeAnimations(output)

import os
import subprocess
import cv2
import imageio.v2 as imageio  # Adjusted import here


from .settings import settings
from .pyplotFunctions import (
    make_images,
    plot_and_save_nodes,
    plot_and_save_mesh,
    plot_and_save_m_mesh,
    plot_and_save_m_diff_mesh,
    plot_and_save_plot,
    plot_and_save_in_poincare_disk,
    plot_and_save_in_e_reduced_poincare_disk,
)
from .dataFunctions import parse_pvd_file, get_data_from_name
from .makePvd import create_collection

from datetime import datetime, timedelta


# This function selects a subset of the vtu files to speed up the animation
# process. (For example, if the video would be 2 hours long, or have a fps of
# 2000, there is no need to use all the frames, so we skip a few)
def select_vtu_files(vtu_files, nrSteps, all_images=False):
    # Always include the first and last frames
    if len(vtu_files) <= 2 or nrSteps <= 2:
        return vtu_files

    # Calculate the step size
    step_size = int(max(1, len(vtu_files) // (nrSteps - 1)))

    # Select files at regular intervals
    if all_images:
        selected_files = vtu_files
    else:
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


def combine_videoes(path, n1, n2, n3=None, n4=None):
    if n3 is None and n4 is None:
        v1 = os.path.join(path, f"{n1}_video.mp4")
        v2 = os.path.join(path, f"{n2}_video.mp4")
        assert os.path.isfile(v1), f"The file {v1} does not exsist"
        assert os.path.isfile(v2), f"The fire {v2} does not exsist"
        # Split the command into a list of arguments to avoid using shell=True
        command = [
            "ffmpeg",
            "-y",  # Automatically overwrite existing file
            "-i",
            v1,
            "-i",
            v2,
            # Filter complex for scaling and cropping to make sure width and height are even
            "-filter_complex",
            "[0:v]scale=-1:1080,crop=iw-mod(iw\\,2):ih-mod(ih\\,2)[v0];"  # Crop if width/height are odd
            "[1:v]scale=-1:1080,crop=iw-mod(iw\\,2):ih-mod(ih\\,2)[v1];"
            "[v0][v1]hstack=inputs=2",
            os.path.join(path, f"{n1}_and_{n2}.mp4"),
        ]
        subprocess.run(command)
    elif n3 is not None and n4 is not None:
        v1 = os.path.join(path, f"{n1}_video.mp4")
        v2 = os.path.join(path, f"{n2}_video.mp4")
        v3 = os.path.join(path, f"{n3}_video.mp4")
        v4 = os.path.join(path, f"{n4}_video.mp4")
        assert os.path.isfile(v1), f"The file {v1} does not exist"
        assert os.path.isfile(v2), f"The file {v2} does not exist"
        assert os.path.isfile(v3), f"The file {v3} does not exist"
        assert os.path.isfile(v4), f"The file {v4} does not exist"

        output_file = os.path.join(path, f"{n1}_{n2}_{n3}_{n4}.mp4")
        dim1 = "1920:1080"
        dim2 = "1920:500"
        filter_complex = (
            f"[0:v]scale={dim1},crop=iw-mod(iw\\,2):ih-mod(ih\\,2)[v0];"
            f"[1:v]scale={dim1},crop=iw-mod(iw\\,2):ih-mod(ih\\,2)[v1];"
            f"[2:v]scale={dim2}:force_original_aspect_ratio=decrease,pad={dim2}:(ow-iw)/2:(oh-ih)/2[v2];"
            f"[3:v]scale={dim2}:force_original_aspect_ratio=decrease,pad={dim2}:(ow-iw)/2:(oh-ih)/2[v3];"
            "[v0][v2]vstack=inputs=2[left];"
            "[v1][v3]vstack=inputs=2[right];"
            "[left][right]hstack=inputs=2"
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            v1,
            "-i",
            v2,
            "-i",
            v3,
            "-i",
            v4,
            "-filter_complex",
            filter_complex,
            output_file,
        ]
        subprocess.run(command)


# Use ffmpeg to convert a folder of .png frames into a mp4 file
def makeAnimations(
    path,
    macroData=None,
    pvdFile=None,
    makeGIF=False,
    transparent=False,
    combineVideos=True,
    useTqdm=True,
    fps=30,
    seconds_per_unit_shear=15,
    allImages=False,
    minTime=7,
    reuseImages=False,
):
    frame_path = os.path.join(path, settings["FRAMEFOLDERPATH"])
    if macroData is None:
        macroData = os.path.join(path, settings["MACRODATANAME"] + ".csv")
    if pvdFile is None:
        pvdFile = os.path.join(path, settings["COLLECTIONNAME"] + ".pvd")

    if not os.path.exists(pvdFile):
        print(f"No file found at: {pvdFile}")
        print("Creating pvd file...")
        create_collection(os.path.join(path, settings["DATAFOLDERPATH"]))

    vtu_files = parse_pvd_file(path, pvdFile)

    # we don't want every frame to be created, so in order to find out what
    # frames should be drawn, we first check how much load change there is
    first = get_data_from_name(vtu_files[0])
    last = get_data_from_name(vtu_files[-1])
    loadChange = float(last["load"]) - float(first["load"])

    # Length of video in seconds
    videoLength = seconds_per_unit_shear * loadChange
    nrSteps = videoLength * fps

    # we select a reduced number of frames
    vtu_files = select_vtu_files(vtu_files, nrSteps, allImages)

    if len(vtu_files) < nrSteps:
        # If we don't have enough frames, we need to make each frame last longer
        # We will make the video last 7 seconds
        fps = len(vtu_files) / minTime

    # Define the path and file name
    # The name of the video is the same as the name of the folder+_video.mp4
    for function, fileName in [
        # (plot_and_save_nodes, "nodes"),
        (plot_and_save_mesh, "mesh"),
        (plot_and_save_in_poincare_disk, "disk"),
        # (plot_and_save_plot, "e_drop_plot"),
        # (plot_and_save_plot, "energy_plot"),
        (plot_and_save_m_diff_mesh, "m_diff_mesh"),
        (plot_and_save_m_mesh, "m_mesh"),
        # (plot_and_save_in_e_reduced_poincare_disk, "erDisk"),
    ]:
        images = make_images(
            vtu_files,
            macro_data=macroData,
            frameFunction=function,
            frame_path=frame_path,
            transparent=transparent,
            use_tqdm=useTqdm,
            reuse_images=reuseImages,
            fileName=fileName,
        )

        # Path to the output video file
        outPath = os.path.join(path, f"{fileName}_video.mp4")

        # Get the last modification time of the last image in the list
        last_image_mod_time = os.path.getmtime(images[-1])

        # Get the last modification time of the output video file
        outPath_mod_time = os.path.getmtime(outPath) if os.path.exists(outPath) else 0

        # Convert modification times to datetime objects
        last_image_mod_datetime = datetime.fromtimestamp(last_image_mod_time)
        outPath_mod_datetime = datetime.fromtimestamp(outPath_mod_time)

        # Calculate the time difference
        time_difference = last_image_mod_datetime - outPath_mod_datetime

        # Check if the time difference is greater than 2 hours
        if time_difference > timedelta(hours=2):
            # If is is larger than two hours, the video was probably not generated with these images

            framesToMp4(images, outPath, fps)
            if makeGIF:
                GIFCommand = [
                    "/opt/homebrew/bin/gifski",
                    "--quality",
                    "100",  # Set to maximum quality
                    "-o",
                    os.path.join(path, f"{fileName}_video.gif"),
                ] + images  # Append the list of image paths to the command
                subprocess.run(GIFCommand)
        else:
            # The video and the last image were generated at about the same time,
            # so the video does not need to be re-rendered
            pass
    if combineVideos:
        try:
            combine_videoes(path, "m_diff_mesh", "m_mesh", "e_drop_plot", "energy_plot")
            combine_videoes(path, "m_mesh", "mesh")
            combine_videoes(path, "mesh", "disk")
            combine_videoes(path, "m_mesh", "disk")
            # combine_videoes(path, "mesh", "erDisk")
        except:
            pass


if __name__ == "__main__":
    pass
    # output = "/Volumes/data/KeepSafe/simpleShear,s150x150l0.15,2e-05,1PBCt4EpsG0.01s0/"

    # # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    # makeAnimations(output)

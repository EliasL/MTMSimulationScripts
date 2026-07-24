import os
import subprocess
import cv2
import imageio.v2 as imageio  # Adjusted import here
from pathlib import Path


from .settings import settings
from .pyplotFunctions import (
    make_images,
    plot_and_save_nodes,
    plot_and_save_mesh,
    plot_and_save_matrix_component_grid,
    plot_and_save_m_mesh,
    plot_and_save_m_diff_mesh,
    plot_and_save_plot,
    plot_and_save_in_poincare_disk,
    plot_and_save_g_in_poincare_disk,
    plot_and_save_in_plastic_reduced_poincare_disk,
    plot_and_save_velocity_field_in_plastic_reduced_poincare_disk,
    plot_and_save_mesh_with_force,
)
from .dataFunctions import resolve_vtu_files, get_data_from_name
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

    try:
        writer = imageio.get_writer(outFile, fps=fps, codec="libx264", quality=7)
        for frame_path in frames:
            frame = imageio.imread(frame_path)
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        print(f"Video writing failed: {e}")


def oldFramesToMp4(frames, outFile, fps):
    print(f"Creating {outFile}")
    # Determine the width and height from the first image
    image_path = frames[0]
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    if not outFile:
        raise ValueError("Output file path (outFile) must not be None or empty.")
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


def combine_videoes(path, n1, n2, n3=None, n4=None, vertical=False):
    if n3 is None and n4 is None:
        v1 = os.path.join(path, f"{n1}_video.mp4")
        v2 = os.path.join(path, f"{n2}_video.mp4")
        assert os.path.isfile(v1), f"The file {v1} does not exist"
        assert os.path.isfile(v2), f"The file {v2} does not exist"

        stack_type = "vstack" if vertical else "hstack"
        scale_filter = "scale=-1:1080" if not vertical else "scale=1920:-1"

        command = [
            "ffmpeg",
            "-y",
            "-i",
            v1,
            "-i",
            v2,
            "-filter_complex",
            f"[0:v]{scale_filter},crop=iw-mod(iw\\,2):ih-mod(ih\\,2)[v0];"
            f"[1:v]{scale_filter},crop=iw-mod(iw\\,2):ih-mod(ih\\,2)[v1];"
            f"[v0][v1]{stack_type}=inputs=2",
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
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
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
    X="load",
    element_subset=None,
    matrix_name="T",
    videoNames=None,
    xlim=None,
    num_processes=-2,
    square_periodic_mesh=False,
    periodic_box_size=None,
    cartesian_viewport_culling=False,
    cartesian_viewport=None,
    output_path=None,
):
    output_path = (
        Path(__file__).resolve().parents[1] / "Plots" / Path(path).name
        if output_path is None
        else Path(output_path)
    )
    output_path.mkdir(parents=True, exist_ok=True)
    frame_path = os.path.join(output_path, settings["FRAMEFOLDERPATH"])
    if macroData is None:
        macroData = os.path.join(path, settings["MACRODATANAME"] + ".csv")
        # Check if file exsists
        if not os.path.isfile(macroData):
            macroData = None
    if pvdFile is None:
        pvdFile = os.path.join(path, settings["COLLECTIONNAME"] + ".pvd")

    if not os.path.exists(pvdFile):
        print(f"No file found at: {pvdFile}")
        print("Creating pvd file...")
        create_collection(os.path.join(path, settings["DATAFOLDERPATH"]))

    vtu_source = pvdFile if os.path.exists(pvdFile) else path
    vtu_files = resolve_vtu_files(vtu_source)
    if xlim is not None:
        xmin, xmax = xlim
        filtered_vtu_files = []
        for vtu_file in vtu_files:
            data = get_data_from_name(vtu_file)
            if X not in data:
                raise ValueError(f"{X} not found in VTU metadata for {vtu_file}")
            x = float(data[X])
            if xmin is not None and x < xmin:
                continue
            if xmax is not None and x > xmax:
                continue
            filtered_vtu_files.append(vtu_file)
        vtu_files = filtered_vtu_files
        if not vtu_files:
            raise ValueError(f"No VTU files found inside {xlim=}")

    # we don't want every frame to be created, so in order to find out what
    # frames should be drawn, we first check how much load change there is
    first = get_data_from_name(vtu_files[0])
    last = get_data_from_name(vtu_files[-1])
    if X in first and X in last:
        xChange = float(last[X]) - float(first[X])
    else:
        xChange = 0.3

    # Length of video in seconds
    videoLength = seconds_per_unit_shear * xChange
    nrSteps = videoLength * fps

    # we select a reduced number of frames
    vtu_files = select_vtu_files(vtu_files, nrSteps, allImages)

    if len(vtu_files) < nrSteps:
        # If we don't have enough frames, we need to make each frame last longer
        # We will make the video last 7 seconds
        fps = len(vtu_files) / minTime

    subset = element_subset
    #subset = "even"
    if isinstance(subset, str):
        subset = subset.strip().lower()
    if subset == "none":
        subset = None

    if matrix_name is None:
        matrix_names = []
    elif isinstance(matrix_name, str):
        name = matrix_name.strip()
        matrix_names = [] if name == "" else [name]
    else:
        matrix_names = list(matrix_name)
        if not all(isinstance(name, str) for name in matrix_names):
            raise TypeError("matrix_name must be a string or an iterable of strings.")
        matrix_names = [name.strip() for name in matrix_names if name.strip()]

    mesh_disk_names = {
        "m_diff_mesh",
        "mesh",
        "m_mesh",
        "mesh_with_forces",
        "disk",
        "disk_G",
        "plasticReductionDisk",
        "plasticReductionDisk_velocity",
    }
    poincare_names = {
        "disk",
        "plasticReductionDisk",
        "plasticReductionDisk_velocity",
    }

    def _is_matrix_component_grid(base_name):
        return base_name.endswith("_component_grid")

    def _with_suffixes(base_name):
        name = base_name
        if base_name in poincare_names:
            tag = "C"
            name = f"{name}_{tag}"
        if square_periodic_mesh and base_name in mesh_disk_names:
            name = f"{name}_square_periodic"
        if subset in ("odd", "even") and (
            base_name in mesh_disk_names or _is_matrix_component_grid(base_name)
        ):
            return f"{name}_{subset}_elements"
        return name

    def _subset_arg(name):
        if subset in ("odd", "even") and (
            name in mesh_disk_names or _is_matrix_component_grid(name)
        ):
            return subset
        return None

    matrix_jobs = [
        (plot_and_save_matrix_component_grid, f"{name}_component_grid", name)
        for name in matrix_names
    ]

    render_jobs = [
        # (plot_and_save_nodes, "nodes"),
        #(plot_and_save_mesh_with_force, "mesh_with_forces"),
        # Move this line to choose where the matrix videos are rendered.
        (plot_and_save_in_poincare_disk, "disk"),
        (plot_and_save_g_in_poincare_disk, "disk_G"),
        (plot_and_save_mesh, "mesh"),
        *matrix_jobs,
        (
            plot_and_save_velocity_field_in_plastic_reduced_poincare_disk,
            "plasticReductionDisk_velocity",
        ),
        (plot_and_save_in_plastic_reduced_poincare_disk, "plasticReductionDisk"),
        (plot_and_save_m_diff_mesh, "m_diff_mesh"),
        (plot_and_save_plot, "energy_plot"),
        (plot_and_save_plot, "e_drop_plot"),
        (plot_and_save_m_mesh, "m_mesh"),
    ]
    if videoNames is not None:
        wanted = {videoNames} if isinstance(videoNames, str) else set(videoNames)
        render_jobs = [
            job
            for job in render_jobs
            if job[1] in wanted or _with_suffixes(job[1]) in wanted
        ]
        if not render_jobs:
            raise ValueError(f"No render jobs matched {videoNames=}")

    # Define the path and file name
    # The name of the video is the same as the name of the folder+_video.mp4
    for job in render_jobs:
        if len(job) == 2:
            function, base_name = job
            matrix_name_for_job = None
        elif len(job) == 3:
            function, base_name, matrix_name_for_job = job
        else:
            raise ValueError(f"Unexpected render job entry: {job}")

        fileName = _with_suffixes(base_name)
        extra_kwargs = {}
        subset_arg = _subset_arg(base_name)
        if matrix_name_for_job is not None:
            extra_kwargs["matrix_name"] = matrix_name_for_job
        images = make_images(
            vtu_files,
            num_processes=num_processes,
            macro_data=macroData,
            frameFunction=function,
            frame_path=frame_path,
            transparent=transparent,
            use_tqdm=useTqdm,
            X=X,
            reuse_images=reuseImages,
            fileName=fileName,
            element_subset=subset_arg,
            square_periodic_mesh=square_periodic_mesh,
            periodic_box_size=periodic_box_size,
            cartesian_viewport_culling=cartesian_viewport_culling,
            cartesian_viewport=cartesian_viewport,
            **extra_kwargs,
        )

        # Path to the output video file
        outPath = os.path.join(output_path, f"{fileName}_video.mp4")
        # Check if the last image is newer than the video
        if not os.path.exists(outPath) or os.path.getmtime(outPath) < os.path.getmtime(
            images[-1]
        ):
            framesToMp4(images, outPath, fps)
            if makeGIF:
                GIFCommand = [
                    "/opt/homebrew/bin/gifski",
                    "--quality",
                    "100",  # Set to maximum quality
                    "-o",
                    os.path.join(output_path, f"{fileName}_video.gif"),
                ] + images  # Append the list of image paths to the command
                subprocess.run(GIFCommand)
        else:
            # The video and the last image were generated at about the same time,
            # so the video does not need to be re-rendered
            pass
    if combineVideos:
        try:
            combine_videoes(
                output_path,
                _with_suffixes("m_diff_mesh"),
                _with_suffixes("mesh"),
                "e_drop_plot",
                "energy_plot",
            )
            # combine_videoes(path, "m_diff_mesh", "m_mesh", "e_drop_plot", "energy_plot")
            combine_videoes(
                output_path, _with_suffixes("mesh"), "energy_plot", vertical=True
            )
            combine_videoes(
                output_path, _with_suffixes("m_mesh"), _with_suffixes("mesh")
            )
            combine_videoes(
                output_path, _with_suffixes("mesh"), _with_suffixes("disk")
            )
            combine_videoes(
                output_path, _with_suffixes("m_mesh"), _with_suffixes("disk")
            )
            combine_videoes(
                output_path,
                _with_suffixes("mesh"),
                _with_suffixes("plasticReductionDisk"),
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    pass
    # output = "/Volumes/data/KeepSafe/simpleShear,s150x150l0.15,2e-05,1PBCt4EpsG0.01s0/"

    # # Replace 'your_pvd_file.pvd' with the path to your .pvd file
    # makeAnimations(output)

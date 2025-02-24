"""
The goal of this script is to reliably see how the MTS2D reacts if one process is
started while another instance is already running and working on the same simulation.

After thinking a bit, i think i'll handle this using slurm instead of trying to
modify the program itself. People (I) will just have to make sure only one simulation
is running at a time.
"""

These scripts are tools to manage the Mesoscopic Tensorial Simulation (MTS)

## Standard power-law energy-drop workflow

When a request says “power-law fit”, “power-law plot”, or asks for the
standard energy-drop result without specifying another population, use this
recipe:

1. Restrict the data to the post-yield region.
2. Extract $\Delta E_R$ and $\Delta E_S$ as paired values from the same
   event transitions, in the same order and with the same length.
3. Use ``simpleDrop`` on the positive $\Delta E_R$ values to define the
   detection/classification threshold $\Delta E_{R,\mathrm{det}}$.
4. Transfer that event classification to the paired $\Delta E_S$ values and
   keep only finite positive $\Delta E_S$ values from irreversible events.
5. Evaluate the KS distance at every observed candidate in that irreversible
   $\Delta E_S$ population and use the true global minimum
   $\Delta E_{S,\min}^{KS}$.  A coarse/local search is an approximation and
   should be selected deliberately.
6. With $\Delta E_{S,\min}^{KS}$ fixed, fit $\alpha$ and $\lambda$
   by maximum likelihood.

The event-pairing contract is represented by ``EventDrops`` in
``Plotting/standardPowerlaw.py``.  Do not independently sort, filter, or
concatenate the two energy-drop arrays before the $\Delta E_R$ classification.



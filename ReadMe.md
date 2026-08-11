These scripts are tools to manage the Mesoscopic Tensorial Simulation (MTS)

## Standard power-law energy-drop workflow

For the simulation energy-drop analysis, the standard recipe is:

1. Restrict the data to the post-yield region.
2. Extract aligned $\Delta E_R$ and $\Delta E_S$ events.
3. Use ``simpleDrop`` on $\Delta E_R$ to define
   $\Delta E_{R,\min}$ and classify reversible and irreversible events.
4. Fit only the irreversible $\Delta E_S$ events.
5. Evaluate the KS distance at every observed candidate
   $\Delta E_{\min}$ in that irreversible population and use the true
   global minimum.  A coarse/local search is an approximation and should be
   selected deliberately.
6. With the selected cutoff fixed, fit $\alpha$ and $\lambda$
   by maximum likelihood.

The low-level power-law functions remain general and allow other populations,
splits, and xmin strategies.  If an analysis uses all events, fits
$\Delta E_R$, uses an Otsu or slope split, or uses a coarse xmin search,
that choice should be explicit and reported as an alternative analysis.  The
same message is included in the command-line help for the power-law scripts.

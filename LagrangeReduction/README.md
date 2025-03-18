

 # Lagrange Reduction Visualization Tool  
 
 This program provides an interactive environment for visualizing the **Lagrange reduction**, in a **configurational space**. It allows users to manipulate basis vectors, and see how the lagrange reduction would recreate the same lattice, but using different basis vectors.
 
 ---
 
 ## Key Features  
 
 ### Vector Manipulation & Volume Constraints  
 - **Drag vectors** to dynamically adjust configurations in the **Lagrange reduction view** or **grid visualization view**.  
 - **Hold `Shift` while dragging** to enable volume constraint. Note that the energy landscape is representative only for **fixed volume**, or an energy density function with a **bulk modulus K=0**. 
 - **Arrow keys** apply **shear transformations** to the vectors:  
   - **Default:** Smooth shear with small steps.  
   - **Hold `Shift`**: Integer shear steps.  
   - **Hold `Alt`**: Finer shear adjustments.  
   - **Hold `Shift + Alt`**: Intermediate shear step size.  
 
 ### Energy and Potential Switching  
 - **Press `S`** to switch to the **square potential** visualization.  
 - **Press `T`** to switch to the **triangular potential** visualization.  
 - **Press `V`** to toggle **volumetric energy visualization**, adjusting energy calculations accordingly.  

 
 ### Visualization & Configuration Modes  
 - **Press `F`** to toggle the **Lagrange reduction visualization**.  
 - **Press `P`** to toggle the **configuration space (Poincaré disk)**.  
 - **Press `V`** to toggle **volumetric energy visualization**, adjusting energy calculations accordingly.  
 - **Press `A`** to toggle the **angle region visualization**, highlighting angular configurations.  
  

 - **Press `B`** to toggle the **background heatmap**, which highlights regions of Lagrange reduction.  
 
 ### Reset & Initialization  
 - **Press `R`** to reset the vectors to their initial state.  
 
 ### Stress, Energy, and Deformation Analysis  
 - Displays **deformation gradients (`F`)**, **Cauchy-Green strain tensors  (`C`)**, and it's **Lagrange reduced version (`C~`)**. 
 - Visualizes **energy fields** across different configurations, with color maps indicating potential energy distributions.  
 - Highlights **reduced configurations** in the **Poincaré disk representation**.  
 - Shows **determinant (`det(F)`)** which turns red when not equal to one. The energy landscape in the Poincare disk is only valid for **`det(F)=1`**. 

 
 ---
 
 This tool provides an intuitive way to explore, manipulate, and analyze the Lagrange reduction.
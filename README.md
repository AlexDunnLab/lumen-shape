# lumen-shape
Supplementary Software for the manuscript: 

Physical basis for the determination of lumen shape in a simple epithelium 

Claudia G. Vasquez, Vipul T. Vachharajani, Carlos O. Garzon Coral, and Alexander R. Dunn, _Nature Communications_ (to appear, 2021)


## Index of files
* lumen_vm.py - Defines simulation objects and methods for vertex-based model of lumen shape.
    * Example usage: 
          
          from lumen_vm import *
          
          
          params = {
                'P_lumen':0.1,
                'k':2,
                'l_a':0.6,
                'l_b':1.5,
                'l_l':0,
                }
          grow_cyst(params,stretched=True)
          

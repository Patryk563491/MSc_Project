# MSc_Project
A repository compiling all the code for the MSc project.

Directories:
  - Data_Preprocessing: contains all the functions for data pre-procesing. Note: requires raw tomograms in MRC format to be processed in Fiji ImageJ and converted to NifTi format.
  - Models/U-Net_Transformer: contains all the U-Net Transformer models used in the project.
  - Training: contains the main_train.py file that was used to train the U-Net and U-Net Transformer models. The other files were used to train the 3D UX-Net.
  - Testing: contains the main_test.py file, which was used to test all of the models. Models were tested individually and then using majority voting.

Patryk Wasniewski
SID: 201338802

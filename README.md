This is the README file for a Generative Adversarial Network in python 3.

Section: Setup

In order to run the code pull the repository to your machine and run:

1. pip3 install keras
2. pip3 install mido
3. pip3 install tqdm
4. pip3 install numpy
5. pip3 install tensorflow

These modules are vital to run the code. 

Section: Running the Software

1. Navigate to the directory that this repository was pulled to.
2. Issue the command python3 delta-extract.py
3. The code will start a Tensorflow backend that will begin training.
4. After each epoch the Network is tested, however it only writes a MIDI 
        every number of cycles defined in the .ini file. Output will be 
        written to the output folder included in the project.

Issues/Known Improvements:

1. All Known Improvements have been implemented at this time. I would like to test on more data and find further ways to improve 
        the algorithm.

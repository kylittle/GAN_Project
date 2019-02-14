This is the README file for a Generative Adversarial Network in python 3.

Section: Setup

In order to run the code pull the repository to your machine and run:

1. pip3 install keras
2. pip3 install mido
3. pip3 install tqdm
4. pip3 install numpy

These modules are vital to run the code. 

Section: Running the Software

1. Navigate to the directory that this repository was pulled to.
2. Issue the command python3 delta-extract.py
3. The code will start a Tensorflow backend that will begin training.
4. After each epoch the Network is tested, however it only writes out a midi
        file every 20 epochs, up to 1000 epochs. It is safe to issue a 
        sigint to stop the programs execution as the last output.mid will be
        in the repositories folder.

Issues/Known Improvements:

1. I want to test the neural network with MIDI files other than Bach to test
        the ability to learn general purpose MIDI files.
2. I would like to add a settings.init file or something that allows easy
        modification of the number of epochs or the frequency of the 
        outputs.

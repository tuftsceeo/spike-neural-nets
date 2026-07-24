Old stuff:
    data_collector.py
    hub_collector.py
    These above two files initially would work together, with one running on the hub and the other running on my computer. Then I updated to Chris's stuff from pyscript and stopped trying to make this work. Was having trouble getting reponse messages from the spike when I send requests.

Real Stuff:
    data_collector_new.py:
        program now used for collecting data. Claud wrote the flairs to make it nicely usable, but uses Hub.py from Hubs to do spike connection
    gesture_data:
        contains all data collected on gestures
    test_spike.py:
        trying to test stuff from spike.py. No longer relevant as spike.py is old and now I use Chris's stuff
    gesture_data:
        contains the gesture data for the four classes. 50 samples per class, each class has 180 data points (6 imu readings x 30 timestamps)
    TennisCNNClassifier: 
        Reads in data from gesture_data and trains model
    TennisPCTest.py:
        Tests the CNNClassifier locally on my PC
    build_hub_cnn.py:
        Trains a cnn model, extracts the weights, and puts into a hub program
    big_cnn_hub_program.py:
        hub program built with extra long floats. did not work on spike (too big)
    cnn_hub_program.py:
        hub program built with shorter floats to overcome the ValueError, but still too big for spike
    


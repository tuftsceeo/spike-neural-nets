four_direction_nn.py: 
    Just the initial version of me figuring out how to create a neural net, 
    not with SPIKE.

These files work together:
DirectionClassifier.py:
    Generates data set, trains model, extracts weights

spike.py:
    Just the SPIKE connection stuff that I had Claud write. Not working yet. 
    For now I am just manually copying the program that main.py writes into 
    the SPIKE app.

main.py:
    Creates a DirectionClassifier, then writes the hub program and connects to 
    SPIKE and uploads program (in theory, not in practice. Does not connect to 
    SPIKE or send program.)

hub_program.py:
    file that main.py writes




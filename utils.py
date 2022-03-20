import os 
import json

import matplotlib.pyplot as plt


def saveSample(sample, data_save_name='hyperparameterSamples.json'):
    path = data_save_name
    
    if not os.path.exists(path):
        with open(path,'w') as f:  # Create file
            f.close()

    with open(path, 'r') as json_file:
        filesize = os.path.getsize(path)

        if filesize == 0:    # If file empty
            jsonFileData = [sample]

        else:
            print(json_file)
            jsonFileData = json.load(json_file)   # Load json data from file

            jsonFileData.append(sample)

    with open(path,'w') as f: 
        json.dump(jsonFileData, f, indent=4)

    json_file.close()
    f.close()

def loadSamples(filename='hyperparameterSamples.json'):
    path = filename
    
    jsonFileData = False
    
    if os.path.exists(path):     # If file exists

        with open(path, 'r') as json_file:
            filesize = os.path.getsize(path)

            if filesize != 0:    # If file not empty
                jsonFileData = json.load(json_file)   # Load json data from file

            json_file.close()

    return jsonFileData


def imshow(inp, title=None):
    """
    Imshow for Tensor.
    Saves to output file output_image.png for use in docker.
    """
    inp = inp.numpy().transpose((1,2,0))
    plt.figure()#figsize = (500,100))
    #plt.figsize=(80, 60)
    if title:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(0.001)    # pause a bit so the plots are updated
    if title:
        plt.savefig(title+".png")
    else:    
        plt.savefig("output_image.png")
    plt.draw()
        
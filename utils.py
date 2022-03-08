import os 
import json


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


        
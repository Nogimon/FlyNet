from glob import glob
import numpy as np
from PIL import Image
from random import shuffle

class DataGenerator:
    def __init__(self, dim_x = 128, dim_y = 128, dim_z = 0, batch_size = 32, shuffle = True, directory = "./nTrain"):
        #Initialization
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.directory = directory

    def getExplorationOrder(self, list_IDs):
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.suffle(indexes)

        return indexes

    def dataGenerateForAllFolders(self, directory):
        while 1:
            folders = sorted(glob(directory + "/*/*/"))
            shuffle(folders)

            train = []
            y = []

            for i in folders:
                #print(i)
                fileo = sorted(glob(i+'*[0-9].png'))
                filem = sorted(glob(i+'*mask.png'))
                #print(len(fileo))
                #print(len(filem))
                for j in range(len(fileo)):
                    img = Image.open(fileo[j])
                    resized = np.asarray(img.resize((128, 128)))

                    imgy = Image.open(filem[j])
                    resizedy = np.asarray(imgy.resize((128, 128)))
                    if(resizedy.max() == 255):
                        resizedy = resizedy / 255
                        #print(i)


                    X = resized[np.newaxis,...,np.newaxis]
                    y = resizedy[np.newaxis,...,np.newaxis]


                    yield (X, y)


    def dataGenerateWithAug(self, directory):
        while 1:
            print("!!!!!Now Starts with a new round!!!!!")
            folders = sorted(glob(directory + "/*/*/"))
            shuffle(folders)

            

            for i in folders:

                

                #print(i)
                fileo = sorted(glob(i+'*[0-9].png'))
                filem = sorted(glob(i+'*mask.png'))
                #print(len(fileo))
                #print(len(filem))
                for j in range(len(fileo)):
                    train = []
                    y = []


                    img = Image.open(fileo[j])
                    resized = np.asarray(img.resize((128, 128)))

                    train.append(resized)

                    augmented = np.rot90(resized)
                    train.append(augmented)

                    augmented = np.rot90(augmented)
                    train.append(augmented)
                
                    augmented = np.rot90(augmented)
                    train.append(augmented)

                    for shift in [5, 10, 20, 30]:
            
                        augmented = np.roll(resized, shift, axis = 1)
                        augmented[:,0:shift] = 0
                        train.append(augmented)
                    
                        augmented = np.roll(resized, shift, axis = 0)
                        augmented[0:shift,:] = 0
                        train.append(augmented)
                    
                        augmented = np.roll(resized, -shift, axis = 1)
                        augmented[:,-shift:] = 0
                        train.append(augmented)
                    
                        augmented = np.roll(resized, -shift, axis = 0)
                        augmented[-shift:,:] = 0
                        train.append(augmented)



                    imgy = Image.open(filem[j])
                    resizedy = np.asarray(imgy.resize((128, 128)))
                    if(resizedy.max() == 255):
                        resizedy = resizedy / 255
                        #print(i)

                    y.append(resizedy)

                    augmented = np.rot90(resizedy)
                    y.append(augmented)

                    augmented = np.rot90(augmented)
                    y.append(augmented)
                
                    augmented = np.rot90(augmented)
                    y.append(augmented)

                    for shift in [5, 10, 20, 30]:
            
                        augmented = np.roll(resizedy, shift, axis = 1)
                        augmented[:,0:shift] = 0
                        y.append(augmented)
                    
                        augmented = np.roll(resizedy, shift, axis = 0)
                        augmented[0:shift,:] = 0
                        y.append(augmented)
                    
                        augmented = np.roll(resizedy, -shift, axis = 1)
                        augmented[:,-shift:] = 0
                        y.append(augmented)
                    
                        augmented = np.roll(resizedy, -shift, axis = 0)
                        augmented[-shift:,:] = 0
                        y.append(augmented)




                    train = np.asarray(train)
                    y = np.asarray(y)
                    train = train[...,np.newaxis]
                    y = y[...,np.newaxis]


                    yield (train, y)
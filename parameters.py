class Parameters:
    directory = "/media/zlab-1/Data/Lian/keras/nData"
    directory2 = "./AD"
    #EP01 : 18-19
    #EP02 : 20-23
    #EP04 : 25-28
    trainstartEP = 31
    trainendEP = 33
    #One good pick: 25-26
    teststartEP = 31
    testendEP = 32
    valEP =[]
    
    
    #Larva01 : 47 - 52
    #Larva02 : 52 - 55
    trainstartLA = 47
    trainendLA = 52
    teststartLA = 50
    testendLA = 51
    valLA =[]
    
    
    trainstartAD = -12
    trainendAD = -10
    teststartAD = -11
    testendAD = -10
    valAD =[]
    
    '''
    folders=sorted(glob(directory+"/*/"))+sorted(glob(directory2+"/*/"))
    folder1=folders[start1:end1]+folders[start2:end2] + folders[start3:end3]
    folder=folders[0:start1]+folders[end1:start2]+folders[end2:start3] + folders[end3:]
    '''

    #y direction um/pixel
    yfactor = 1.12 * 200 / 128
    #x direction um/pixel
    xfactor = 2.2
    #time frames/second
    timefactor = 129.9

    #The 'good' numbers are vali:8, test:9
    valifolder = 8
    testfolder = 9

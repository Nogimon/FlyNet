class Parameters:
    directory = "/media/zlab-1/Data/Lian/keras/nData"
    directory2 = "./AD"
    #EP01 : 18-19
    #EP02 : 20-23
    #EP04 : 25-28
    trainstartEP = 25
    trainendEP = 28
    #One good pick: 25-26
    teststartEP = 25
    testendEP = 26
    
    #Larva01 : 47 - 52
    #Larva02 : 52 - 55
    trainstartLA = 47
    trainendLA = 52
    teststartLA = 50
    testendLA = 51
    
    trainstartAD = -12
    trainendAD = -10
    teststartAD = -11
    testendAD = -10
    '''
    folders=sorted(glob(directory+"/*/"))+sorted(glob(directory2+"/*/"))
    folder1=folders[start1:end1]+folders[start2:end2] + folders[start3:end3]
    folder=folders[0:start1]+folders[end1:start2]+folders[end2:start3] + folders[end3:]
    '''
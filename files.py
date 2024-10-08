import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
        
        
def deletFolder(directory):
    import shutil
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror)) 
    

def delet_files(directory,ftype):
    filelist = [ f for f in os.listdir(directory) if f.endswith(ftype) ]
    for f in filelist:
        os.remove(os.path.join(directory, f))





    
    
import os 
from pathlib import Path
import logging

#doing the lof file configuration :
                                                #current time  #with the logging message
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')


project_name= "cnnClassfier"

#list of files to be created :

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
     "config.yaml",
     "dvc.yaml",
     "params.yaml",
     "requiremnts.txt",
     "setup.py",
     "research/trials.ipynb"


    






]

for file_path in list_of_files:
    file_path=Path(file_path)#checking the apath type to path according ot os 
    filedir,filename=os.path.split(file_path) #splitting the file path and file dir
       

    if filedir !="":
       os.makedirs(filedir,exist_ok=True)
       logging.info(f"creating Directory;{filedir} for the file:{filename} ")
     
    if (not os.path.exists((file_path))) or (os.path.getsiz(file_path)==0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"created empy file :{file_path}") 
    else:
        logging.info(f"{filename} is already exist")
        


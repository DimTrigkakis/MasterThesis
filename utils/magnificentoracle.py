from __future__ import print_function
import os
from pprint import pprint
import torchvision.transforms as transforms
import random
import torch 
from PIL import Image 

class MagnificentOracle(object):
    
    level_description = {"STO":"Standard Output","INFO":"Log Info"}

    def __init__(self):
        self.level = "STO"
        self.omit = False
        self.logfile = None

    def set_omit(self, omit = False):
        self.omit = omit

    def sef_level(self, level = "STO"):
        self.level = level

    def set_log(self, logfile = None):
        self.logfile = logfile
        self.level = "INFO"

        if self.logfile != None:
            try:
                os.remove(self.logfile)
            except OSError:
                pass


        else:
            self.level = "STO"

    def log(self, object):

        if isinstance(object, str):
            if object == "-dotted-line":
                object = "-------------------------------------------------------------"
            # expand section

        text = "Logger [{}] -->".format(self.level_description[self.level])

        if self.omit:
            text = ""

        if self.level == "STO":
            print(text,str(object))
        else:
            if self.logfile == None:
                print(text,object)
            else:
                mymode = "w"
                if os.path.exists(self.logfile):
                    mymode = "a+"
                with open(self.logfile,mymode) as f:
                    f.write(text+str(object)+"\n")

    def visual_tensor(self, tensor, reverse_transform = None, figure_name = "./figures/visuatensor"):

        '''Pillow image conversion of a tensor object.

        Arguments:
            tensor (Tensor) : The cpu tensor object to convert into an image

        Returns a Pillow image object.
        '''

        cpu_tensor = tensor.cpu()
        shape = cpu_tensor.size()
        
        if len(shape) == 4:
            if shape[1] == 3:
                if reverse_transform == None:
                    unnormalize = transforms.Normalize(mean=[-0.411/0.170, -0.493/0.178,-0.309/0.171],std=[1/0.170, 1/0.178, 1/0.171])
                    topil = transforms.ToPILImage()
                    chunk_of_tensor = unnormalize(cpu_tensor[0])
                    img = topil(chunk_of_tensor)
                else:
                    # not tested
                    chunk_of_tensor = reverse_transform(cpu_tensor[0])
                    chunk_of_tensor = torch.clamp(chunk_of_tensor,0,1)
                    img = topil(chunk_of_tensor)

        img.save(figure_name+".jpg")
        img.close()

        return img









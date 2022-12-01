import torch
import numpy as np
import cv2
from fit_fncs import FITNESS_FNS
import copy

## fitness function coordinate system:
#        -1
#        |
#        |
# -1 --------- 1 (+)
#        |
#        |
#        1 (+)


## render coordinate:
# 0,0----> +
# |
# |
# v 
# +
class Fitness_Field:

    def __init__(self, fitness_fnc, start_pos, inp_range = (-1,1), render_size=(256,256)):
        self.fitness_fnc = fitness_fnc
        self.inp_range = inp_range
        self.render_size = render_size
        self.start_pos = np.array(start_pos)
        self.pos = copy.deepcopy(self.start_pos)
        self.traj = []

        with torch.no_grad():
            self.fitness_img = self.create_fitness_img()

    def create_fitness_img(self):
        inp_min, inp_max = self.inp_range
        x = torch.linspace(inp_min, inp_max, self.render_size[0])
        y = torch.linspace(inp_min, inp_max, self.render_size[0])
        x, y = torch.meshgrid(x,y)
        coords = torch.cat((y.reshape(-1,1), x.reshape(-1,1)), axis=-1)
        
        img = self.fitness_fnc(coords).reshape(x.shape)
        img = img.numpy()
        img = img - img.min()
        img = img if img.shape[-1] == 3 else np.stack([img]*3, axis=-1)
        img = (255*img/img.max()).astype(np.uint8)
        return img

    def get_fitness_img(self):
        return copy.deepcopy(self.fitness_img)

    def reset(self):
        ## Fitness function ---> no update

        return None, None

    def step(self, pos, ret_numpy = True):
        """
        input:
            pos \in [-1,1]
        """
        if type(pos) != torch.Tensor:
            self.traj.append(pos)
            self.pos = pos
            pos = torch.from_numpy(pos)
        else:
            self.pos = pos.detach().numpy()
            self.traj.append(pos.detach().numpy())
        
        if ret_numpy:
            return None, self.fitness_fnc(pos).detach().numpy(), True, False, {}
        else:
            return None, self.fitness_fnc(pos), True, False, {}
    
    
    def fix_coords(self, pos):
        pos = (pos + 1)/2 * self.render_size[0]
        return np.floor(np.clip(pos,0,self.render_size[0] - 1)).astype(int)
        
    def render(self, img = None, pos = None):
        pos = pos if pos is not None else self.pos
        col = (0,255,0)
        img = copy.deepcopy(self.fitness_img) if img is None else img

        return cv2.circle(img, self.fix_coords(pos).squeeze(), 4, col, -1)


def make_env(fn_type = "parabola"):
    return Fitness_Field(FITNESS_FNS[fn_type]["fitness_fnc"],
                         FITNESS_FNS[fn_type]["start_pos"])

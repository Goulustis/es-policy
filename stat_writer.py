import cv2
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt
import os.path as osp

class Renderer:
    def __init__(self, env, policy, path = None, config=None):
        self.env = env
        self.policy = policy
        self.config=config

        self.path = "dev.mp4" if path is None else path 
        self.pops = None 
        self.recorded_frames = []
        self.fps = 10

        self.es_norms = []
        self.fd_norms = []
        self.rewards = []

    def update_log(self, log):
        self.update_pop(log.get("pop"))
        self.update_norms(log)
        self.update_rewards(log)

    def update_rewards(self, log):
        if log.get("rewards") is not None:
            self.rewards.append(log.get("rewards"))

    def update_pop(self, pop):
        if self.pops is None:
            self.pops = [pop] 
        else:
            self.pops.append(pop)
    
    def update_norms(self, log):
        if log.get("es_norm") is not None:
            self.es_norms.append(log["es_norm"])
        
        if log.get("fd_norm") is not None:
            self.fd_norms.append(log["fd_norm"])
    
    # def draw_pop(self, pop, frame, train_info =None):
    #     col = (255,0,0)
    #     if pop is None:
    #         return
    #     for i, pnt in enumerate(pop):
    #         if self.config["optim_method"] == "es_grad":
    #             if train_info is not None and \
    #                train_info.get("msk_val") is not None: 
    #                 tmp = np.zeros_like(frame, np.uint8)
    #                 cv2.circle(tmp,
    #                                 self.env.fix_coords(pnt).squeeze(),
    #                                 2, # radius 
    #                                 col, cv2.FILLED)
                    
    #                 msk = tmp[...,0].astype(bool)
    #                 alpha = train_info["msk_val"]
    #                 frame[msk] = cv2.addWeighted(frame, alpha, tmp, 1-alpha, 0)[msk]
    #         else:
    #             cv2.circle(frame, self.env.fix_coords(pnt).squeeze(), 2, col, -1)
            
    #     return frame
    def draw_pop(self, pop, frame, train_info =None):
        col = (255,0,0)
        if pop is None:
            return
        for i, pnt in enumerate(pop):
            if self.config["optim_method"] == "es_grad":
                if train_info is not None and \
                   train_info.get("msk_val") is not None and \
                   train_info["msk_val"] <= 0.5: 
                    cv2.circle(frame, self.env.fix_coords(pnt).squeeze(), 2, col, -1)
            else:
                cv2.circle(frame, self.env.fix_coords(pnt).squeeze(), 2, col, -1)
            
        return frame


    def capture_frame(self, train_info = None):
        frame = self.env.get_fitness_img()
        if self.pops is not None:
            frame = self.draw_pop(self.pops[-1], frame, train_info)
        self.recorded_frames.append(self.env.render(frame, self.policy.get_pos()))
    
    def close(self):
        clip = ImageSequenceClip(self.recorded_frames, fps=self.fps)
        clip.write_videofile(self.path)
        
        if len(self.es_norms) != 0:
            plt.clf()
            plt.plot(self.es_norms, label = "es norm")
            plt.plot(self.fd_norms, label = "fd norm")
            plt.legend()
            plt.savefig(osp.join(osp.dirname(self.path),"grad_norm.png"))

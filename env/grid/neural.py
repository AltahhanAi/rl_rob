from env.grid.tabular import *
import cv2
#============================================================================================
#======================Defining a Gridi: A Grid with Images States===========================
''' 
    Below, we establish a class that will return after each step an observation which is the 
    image of the grid instead of its state id. This is essential to be able to deal with a 
    more general set of RL methods that are capable of learning by observing its environment 
    instead of being given the state's id. 

    Important: this class gives an image with dims of
    (C,H,W) expected for PyTorch, while
    (H,W,C) expected TensorFlow. So, if you want to use it with TensorFlow, set 
    self.img = np.expand_dims(self.img, -1)
'''

import cv2
import numpy as np
# ======================================================================================================================
'''
This class is lightweight; it uses the agent's state, goals, and obstacles' coordinates to generate a greyscale pixel directly, rather than rendering a plot and then storing its values.
'''
class iGrid(Grid):
    def __init__(self, resize=True, size=(50, 84), flatten=False, **kw):
        super().__init__(**kw)
        self.resize = resize
        self.size = size
        self.flatten = flatten

        self.bg_val = 0.0
        self.obs_val = 0.35
        self.cliff_val = 0.55
        self.goal1_val = 0.75
        self.goal2_val = 0.9
        self.agent_val = 1.0

    def to_img(self):
        def to_r_c(s): 
            return s // self.cols, s % self.cols
            
        img = np.full((self.rows, self.cols), self.bg_val, dtype=np.float32)

        for s in self.obstacles:  img[to_r_c(s)] = self.obs_val
        for s in self.cliffs:     img[to_r_c(s)] = self.cliff_val

        img[to_r_c(self.goals[0])] = self.goal1_val
        img[to_r_c(self.goals[1])] = self.goal2_val
        img[to_r_c(self.s)] = self.agent_val

        if self.resize:
            img = cv2.resize( img, dsize=(self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST )
        
        if self.flatten: return img.reshape(-1) 
        else:            return np.expand_dims(img, 0)   # (C,H,W) as in pytorch change to np.expand_dims(img, -1) for TensorFlow 

    def s_(self):
        return self.to_img() 

    def S_(self):
        sc = self.s
        imgs = []
        for self.s in range(self.nS):
            imgs.append(self.s_())  # each is (1, H, W)
        self.s = sc
        return np.stack(imgs)  # (nS, 1, H, W)
    
# =========================================================================================================================
'''
This class is heavier and slower because it uses images from matplotlib, which was not specifically designed for fast, step-wise image generation. This is for illustration and testing purposes. It is useful to know if you set up a neural net solution for RL correctly.
    
Note that we save the images in a folder called img, so you will need to create such a folder in this notebook's folder. If you want to change this behaviour or save images in the same folder of this notebook, adjust the code accordingly.
 
'''
class i_Grid(Grid):
    
    def __init__(self, animate=False, saveimg=False, resize=True, size=(50,84), **kw):
        super().__init__(**kw)
        self.i = 0                  # snapshot counter
        self.img = None             # snapshot image
        self.io = None              # snapshot io buffer
        self.animate = animate
        self.saveimg = saveimg
        self.resize = resize
        self.size = size 
        

    def render_image(self, saveimg):
       # prepare and scale the area that will be captured 
        if self.io is None: self.io = io.BytesIO()
        
        # scale = 0.0138888 # use this if you are using Jupyter notebooks
        scale = 0.01      # use this if you are using Jupyter Lab
        box = self.ax0.get_window_extent().transformed(mtransforms.Affine2D().scale(scale))
        
        # place frame in memory buffer then save to disk if you want
        plt.savefig(self.io, format='raw', bbox_inches=box)
        if saveimg or self.img is None:
            os.makedirs('img') if not os.path.exists('img') else None
            plt.savefig('img/img%d.png'%self.i, bbox_inches=box); self.i+=1 
            if self.img is None: 
                self.newshape = plt.imread('img/img0.png').shape
        #try:
        # reshape the image and store the current image 
        self.img = np.reshape(np.frombuffer(self.io.getvalue(),dtype=np.uint8),newshape=self.newshape)[:,:,:3]
        #except:
            #self.img = np.frombuffer(self.io.getvalue(), dtype=np.uint8)[:,:,:3]
            #print('could not convert the image')
        if self.resize:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.img = cv2.resize(self.img , dsize=(self.size[1],self.size[0]), interpolation=cv2.INTER_CUBIC)/255
            self.img = np.expand_dims(self.img, 0)# slightly better than  self.img = self.img[:,:,np.newaxis]
        # save only the latest image to the buffer and not accumulate
        self.io.seek(0) 

    # note that when we visualise render__() will be called twice, which is a bit inefficient, but is kept for simplicity
    def s_(self):
        self.render__(image=True, animate=self.animate, saveimg=self.saveimg)#, animate=self.animate)
        return self.img

def irandwalk(**kw):  return randwalk  (iGrid, **kw)
def irandwalk_(**kw): return randwalk_ (iGrid, **kw)
def igrid(**kw):      return grid      (iGrid, **kw)
def imaze(**kw):      return maze      (iGrid, **kw)
def imaze_large(**kw):return maze_large(iGrid, **kw)
def icliffwalk(**kw): return cliffwalk (iGrid, **kw)
def iwindy(**kw):     return windy     (iGrid, **kw)


# def imaze(Grid=iGrid, r=6, c=9, **kw): # we cover this later
#     return iGrid(gridsize=[r,c], s0=r//2*c, goals=[r*c-1], style='maze', **kw)#figsize is made ineffective

def i_maze(Grid=i_Grid, r=6, c=9, **kw): # we cover this later
    return iGrid(gridsize=[r,c], s0=r//2*c, goals=[r*c-1], style='maze', **kw)#figsize is made ineffective

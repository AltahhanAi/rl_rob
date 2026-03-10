#=================================================================================================

'''
    This library handles a set of Grid worlds and their visualisation.
    Below, we provide a series of Grid classes that build on top of each other, 
    each time we add a bit more functionality. 
    
    You **do not** need to study or understand the code; 
    You need to know how to use the Grid() class, as explained in Lesson 3.

    Enjoy!
'''

#=================================================================================================
'''
    imports
'''
import numpy as np
import time
import io
import os
import random

from IPython.display import clear_output, display, HTML
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import colors

from numpy.random import rand, seed, randint, choice
from random import choices, sample
from tqdm import trange, tqdm
from numbers import Integral

# ================================================================================================================
# useful variables to be able to deal with actions using var names, note that we must keep the order consistent 
left, right, down, up, lef_down, lef_up, right_down, right_up = tuple(range(8))

class π_To_i(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.type = Integral if all(isinstance(k, Integral) for k in self.keys()) else str

    def __getitem__(self, π):
        g = dict.__getitem__
        f = lambda a: g(self, a) if isinstance(a, self.type) and a in self else a
        seq = (not isinstance(π, (str, bytes))) and hasattr(π, "__iter__")
        return list(map(f, π.tolist() if hasattr(π, "tolist") else π)) if seq else f(π)
        
πTi = π_To_i({'←':0, '→':1, '↓':2, '↑':3, '↙':4, '↖':5, '↘':6, '↗':7})
iTπ = π_To_i({0:'←', 1:'→', 2:'↓', 3:'↑', 4:'↙', 5:'↖', 6:'↘', 7:'↗'})

# =================================simple Grid world (no visualisation)==========================================

'''
Actions and States
    If the action is right, we add 1; if it is left, we subtract 1. 
    When we want the agent to move up, we add one row _number of cols in the grid_, 
    and when we want the agent to move down, we subtract cols. Moving diagonally is similar, 
    but we need to combine moving up and left, down and left, up and right, and down and right. 
    Checking if the move is allowed (new state is allowed) as per the agent's current state 
    consists of checking for 4 cases. 
    1. the agent is not bypassing outside the grid to boundaries (top or bottom)
    2. the agent is not bypassing the right edge of the 2-d grid
    3. the agent is not bypassing the left edge of the 2-d grid
    4. the agent is not stepping into an obstacle

Rewards
    The set of possible rewards is 4; one for 2 possible goals (terminal states), 
    one for intermediate (non-terminal) states and one special for cliff falling. 
    The class caller can assign one of the special rewards by passing its name, 
    and the setattr() function will handle the rest.


'''
class Grid:
    def __init__(self, gridsize=[6,9], nA=4, s0=3*9, goals=None, Vstar=None):
        self.rows = gridsize[0]
        self.cols = gridsize[1]
        self.nS = self.cols*self.rows # we assume cell IDs go left to right and top down
        self.goals = [self.nS-1, self.nS-1] if goals is None else ([goals[0], goals[0]] if len(goals)==1 else goals)
        self.S = [s for s in range(self.nS) if s not in self.goals] # set of nonterminal states S
        self.S_= [s for s in range(self.nS)]                   # set of all states         S+
        
        self.Vstar = Vstar # optimal state value, needed for some of the environments
        self.s0 = s0
        self.s = s0
        self.trace = [self.s0]
        
        # actions ---------------------------------------------------------------------       
        cols = self.cols
        self.actions_2 = [-1, +1]
        self.actions_4 = [-1, +1, -cols, +cols]     # left, right, down and up
        self.actions_8 = [-1, +1, -cols, +cols, -1-cols, -1+cols, 1-cols, 1+cols] # left-down, left-up, right-down, right-up

        self.nA = nA
        if nA==2: self.actions = self.actions_2
        if nA==4: self.actions = self.actions_4
        if nA==8: self.actions = self.actions_8
        
        # rewards types-----------------------------------------------------------------
        self.nR = 4
        self.rewards = [0, 1, 0, -100] # intermediate, goal1, goal2, cliff
        self.obstacles, self.cliffs = [], [] # lists that will be checked when doing actions
        
        
    def reset(self, withtrace=True):
        self.s = self.s0
        if withtrace: self.trace = [self.s0]
        return self.s_()
    #-----------------------------------------rewards related-------------------------------------------
    def rewards_set(self):
        return np.array(list(set(self.rewards)))
        
    def reward(self,s=None):
        stype = self.stype(s)
        reward = self.rewards[stype]
        if stype==3: self.reset(False)    # s in cliffs
        return reward, 2>=stype>=1        # either at goal1 or goal2
    
    #-----------------------------------------actions related-------------------------------------------
    def invalid(self, s,a):
        cols = self.cols
        # invalid moves are 
        # 1. off grid boundaries
        # 2. off the right edge (last and is for right up and down diagonal actions)
        # 3. off the left edge  (last and is for left  up and down diagonal actions)
        # 4. into an obstacle
        return      not(0<=(s+a)<self.nS) \
                    or (s%cols!=0 and (s+a)%cols==0 and (a==1 or a==cols+1 or a==-cols+1))  \
                    or (s%cols==0 and (s+a)%cols!=0 and (a==-1 or a==cols-1 or a==-cols-1)) \
                    or (s+a) in self.obstacles

    def step(self, a, *args):
        a = self.actions[a]
        if not self.invalid(self.s,a): self.s += a
        
        self.trace.append(self.s)
        reward, done = self.reward()           # must be done in this order for the cliff reset to work properly
        return self.s_(), reward, done, {}, {} # empty dict for compatibility
    
    #-----------------------------------------state related-------------------------------------------
    # useful for inheritance, observation can be a state (index) or a state representation (vector or image)
    def s_(self):
        return self.s
    
    # returns the number of states that are available for the agent to occupy
    def nS_available(self):
        return self.nS - len(self.obstacles)
    
    #-----------------------------------------goals related-------------------------------------------
    # returns the type of the current state (0: intermediate, 1 or 2 at goal1 or goal2, 3:off cliff)
    def stype(self,s=None):
        s = self.s if s is None else s
        goals, cliffs = self.goals, self.cliffs
        # the order is significant and must not be changed
        return [s not in goals+cliffs, s==goals[0], s==goals[1], s in cliffs].index(True)
    
    def isatgoal(self):
        return self.stype() in [1,2] # either at goal1 or goal2


#=================================================================================================
#===========================differen Grid world styles no visualisation===========================

'''
    In all of our treatments, we will follow a trend of using the same class name 
    if possible for the child and parent class, so that we do not need to deal 
    with different class names when we import these classes. So Grid(Grid),
    means the new Gris class inherits from the previous Grid class. This allows us 
    to gradually build the capabilities of our classes in a concise and manageable 
    manner. 
    
    The getattr(self, reward) function allows us to pass a string to the class setter 
    ex. Grid(reward='cliffwalk') and then Python will search and return a corresponding 
    attribute or function with the same name ex. self.cliffwalk.

    Finally, due to the wind adding to the displacement of the robot, we had to override 
    the step(a) function. The function checks the validity of an action and then attempts 
    to add as much of the wind displacement as the grid boundaries allow.
'''
class Grid(Grid):
    def __init__(self, reward='',  style='', **kw):
        super().__init__(**kw)
    
        # explicit rewards for[intermediate,goal0,goal1, cliff] states
        self.reward_    = [0,    1,   0, -100] # this is the default value for the rewards
        self.cliffwalk  = [-1,  -1,  -1, -100]
        self.randwalk   = [ 0,   0,   1,    0]
        self.randwalk_  = [ 0,  -1,   1,    0]
        self.reward0    = [-1,   0,   0,   -1]
        self.reward_1   = [-1,  -1,  -1,   -1]
        self.reward1    = [-1,   1,   1,   -1]
        self.reward10   = [-1,  10,  10,   -1]
        self.reward100  = [-1, 100, 100,   -1]
        self.rewardall1 = [ 1,   1,   1,    1]
        

        if reward: self.rewards  = getattr(self, reward)
        self.style = style
        
        # accommodating grids styles -------------------------------------------------------------
        self.X, self.Y = None, None
        self.Obstacles = self.Cliffs = 0 # np arrays for display only, related to self.obstacles, self.cliffs
        self.wind = [0]*10               # [0,0,0,0,0,0,0,0,0,0]
        
        if self.style=='cliff':
            self.Cliffs = None           # for displaying only, to be filled when render() is called
            self.cliffs = list(range(1,self.cols-1))
            
        elif self.style=='maze':
            self.Obstacles = None        # for displaying only, to be filled when render() is called
            rows = self.rows
            cols = self.cols
            # midc = int(cols/2)
            obstacles1 = list(range(2+2*cols, 2+(rows-1)*cols, cols))    # set of vertical obstacles near the start
            obstacles2 = list(range(5+cols, 2*cols-3))                   # set of horizontal obstacles
            obstacles3 = list(range(-2+4*cols,-2+(rows+1)*cols, cols))   # set of vertical obstacles near the end
            self.obstacles = obstacles1 + obstacles2 + obstacles3        # concatenate them all 

        # upward winds intensity for each column
        elif self.style=='windy':
            self.wind = [0,0,0,1,1,1,2,2,1,0] # as in example 6.5 of the book
    
    # override the step() function so that it can deal with wind
    def step(self, a, *args):
        a = self.actions[a]
        if not self.invalid(self.s,a): self.s += a
        
        if self.style=='windy':
            maxwind = self.wind[self.s%self.cols]
            for wind in range(maxwind, 0, -1): # we need to try apply all the wind or at least part of it
                if not self.invalid(self.s, wind*self.cols): self.s += wind*self.cols; break
        
        self.trace.append(self.s)
        reward, done = self.reward()       # must be done in this order for the cliff reset to work properly
        return self.s_(), reward, done, {}, {} # empty dict for compatibility

from matplotlib import colors

from numpy.random import rand, seed, randint, choice
from random import choices, sample
from tqdm import trange, tqdm
from numbers import Integral

# ================================================================================================================
# useful variables to be able to deal with actions using var names, note that we must keep the order consistent 
left, right, down, up, lef_down, lef_up, right_down, right_up = tuple(range(8))

class π_To_i(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.type = Integral if all(isinstance(k, Integral) for k in self.keys()) else str

    def __getitem__(self, π):
        g = dict.__getitem__
        f = lambda a: g(self, a) if isinstance(a, self.type) and a in self else a
        seq = (not isinstance(π, (str, bytes))) and hasattr(π, "__iter__")
        return list(map(f, π.tolist() if hasattr(π, "tolist") else π)) if seq else f(π)
        
πTi = π_To_i({'←':0, '→':1, '↓':2, '↑':3, '↙':4, '↖':5, '↘':6, '↗':7})
iTπ = π_To_i({0:'←', 1:'→', 2:'↓', 3:'↑', 4:'↙', 5:'↖', 6:'↘', 7:'↗'})

# =================================simple Grid world (no visualisation)==========================================

'''
Actions and States
    If the action is right, we add 1; if it is left, we subtract 1. 
    When we want the agent to move up, we add one row _number of cols in the grid_, 
    and when we want the agent to move down, we subtract cols. Moving diagonally is similar, 
    but we need to combine moving up and left, down and left, up and right, and down and right. 
    Checking if the move is allowed (new state is allowed) as per the agent's current state 
    consists of checking for 4 cases. 
    1. the agent is not bypassing outside the grid to boundaries (top or bottom)
    2. the agent is not bypassing the right edge of the 2-d grid
    3. the agent is not bypassing the left edge of the 2-d grid
    4. the agent is not stepping into an obstacle

Rewards
    The set of possible rewards is 4; one for 2 possible goals (terminal states), 
    one for intermediate (non-terminal) states and one special for cliff falling. 
    The class caller can assign one of the special rewards by passing its name, 
    and the setattr() function will handle the rest.


'''
class Grid:
    def __init__(self, gridsize=[6,9], nA=4, s0=3*9, goals=None, Vstar=None):
        self.rows = gridsize[0]
        self.cols = gridsize[1]
        self.nS = self.cols*self.rows # we assume cell IDs go left to right and top down
        self.goals = [self.nS-1, self.nS-1] if goals is None else ([goals[0], goals[0]] if len(goals)==1 else goals)
        self.S = [s for s in range(self.nS) if s not in self.goals] # set of nonterminal states S
        self.S_= [s for s in range(self.nS)]                   # set of all states         S+
        
        self.Vstar = Vstar # optimal state value, needed for some of the environments
        self.s0 = s0
        self.s = s0
        self.trace = [self.s0]
        
        # actions ---------------------------------------------------------------------       
        cols = self.cols
        self.actions_2 = [-1, +1]
        self.actions_4 = [-1, +1, -cols, +cols]     # left, right, down and up
        self.actions_8 = [-1, +1, -cols, +cols, -1-cols, -1+cols, 1-cols, 1+cols] # left-down, left-up, right-down, right-up

        self.nA = nA
        if nA==2: self.actions = self.actions_2
        if nA==4: self.actions = self.actions_4
        if nA==8: self.actions = self.actions_8
        
        # rewards types-----------------------------------------------------------------
        self.nR = 4
        self.rewards = [0, 1, 0, -100] # intermediate, goal1, goal2, cliff
        self.obstacles, self.cliffs = [], [] # lists that will be checked when doing actions
        
        
    def reset(self, withtrace=True):
        self.s = self.s0
        if withtrace: self.trace = [self.s0]
        return self.s_()
    #-----------------------------------------rewards related-------------------------------------------
    def rewards_set(self):
        return np.array(list(set(self.rewards)))
        
    def reward(self,s=None):
        stype = self.stype(s)
        reward = self.rewards[stype]
        if stype==3: self.reset(False)    # s in cliffs
        return reward, 2>=stype>=1        # either at goal1 or goal2
    
    #-----------------------------------------actions related-------------------------------------------
    def invalid(self, s,a):
        cols = self.cols
        # invalid moves are 
        # 1. off grid boundaries
        # 2. off the right edge (last and is for right up and down diagonal actions)
        # 3. off the left edge  (last and is for left  up and down diagonal actions)
        # 4. into an obstacle
        return      not(0<=(s+a)<self.nS) \
                    or (s%cols!=0 and (s+a)%cols==0 and (a==1 or a==cols+1 or a==-cols+1))  \
                    or (s%cols==0 and (s+a)%cols!=0 and (a==-1 or a==cols-1 or a==-cols-1)) \
                    or (s+a) in self.obstacles

    def step(self, a, *args):
        a = self.actions[a]
        if not self.invalid(self.s,a): self.s += a
        
        self.trace.append(self.s)
        reward, done = self.reward()           # must be done in this order for the cliff reset to work properly
        return self.s_(), reward, done, {}, {} # empty dict for compatibility
    
    #-----------------------------------------state related-------------------------------------------
    # useful for inheritance, observation can be a state (index) or a state representation (vector or image)
    def s_(self):
        return self.s
    
    # returns the number of states that are available for the agent to occupy
    def nS_available(self):
        return self.nS - len(self.obstacles)
    
    #-----------------------------------------goals related-------------------------------------------
    # returns the type of the current state (0: intermediate, 1 or 2 at goal1 or goal2, 3:off cliff)
    def stype(self,s=None):
        s = self.s if s is None else s
        goals, cliffs = self.goals, self.cliffs
        # the order is significant and must not be changed
        return [s not in goals+cliffs, s==goals[0], s==goals[1], s in cliffs].index(True)
    
    def isatgoal(self):
        return self.stype() in [1,2] # either at goal1 or goal2


#=================================================================================================
#===========================differen Grid world styles no visualisation===========================

'''
    In all of our treatments, we will follow a trend of using the same class name 
    if possible for the child and parent class, so that we do not need to deal 
    with different class names when we import these classes. So Grid(Grid),
    means the new Gris class inherits from the previous Grid class. This allows us 
    to gradually build the capabilities of our classes in a concise and manageable 
    manner. 
    
    The getattr(self, reward) function allows us to pass a string to the class setter 
    ex. Grid(reward='cliffwalk') and then Python will search and return a corresponding 
    attribute or function with the same name ex. self.cliffwalk.

    Finally, due to the wind adding to the displacement of the robot, we had to override 
    the step(a) function. The function checks the validity of an action and then attempts 
    to add as much of the wind displacement as the grid boundaries allow.
'''
class Grid(Grid):
    def __init__(self, reward='',  style='', **kw):
        super().__init__(**kw)
    
        # explicit rewards[intermediate, goal0, goal1, cliff] states
        self.reward_    = [0,    1,   0, -100] # this is the default value for the rewards
        self.cliffwalk  = [-1,  -1,  -1, -100]
        self.randwalk   = [ 0,   0,   1,    0]
        self.randwalk_  = [ 0,  -1,   1,    0]
        self.reward0    = [-1,   0,   0,   -1]
        self.reward_1   = [-1,  -1,  -1,   -1]
        self.reward1    = [-1,   1,   1,   -1]
        self.reward10   = [-1,  10,  10,   -1]
        self.reward100  = [-1, 100, 100,   -1]
        self.rewardall1 = [ 1,   1,   1,    1]
        

        if reward: 
            if isinstance(reward, str): self.rewards  = getattr(self, reward)
            else:                       self.rewards  = reward
        self.style = style
        
        # accommodating grids styles -------------------------------------------------------------
        self.X, self.Y = None, None
        self.Obstacles = self.Cliffs = 0 # np arrays for display only, related to self.obstacles, self.cliffs
        self.wind = [0]*10               # [0,0,0,0,0,0,0,0,0,0]
        
        if self.style=='cliff':
            self.Cliffs = None           # for displaying only, to be filled when render() is called
            self.cliffs = list(range(1,self.cols-1))
            
        elif self.style=='maze':
            self.Obstacles = None        # for displaying only, to be filled when render() is called
            rows = self.rows
            cols = self.cols
            # midc = int(cols/2)
            obstacles1 = list(range(2+2*cols, 2+(rows-1)*cols, cols))    # set of vertical obstacles near the start
            obstacles2 = list(range(5+cols, 2*cols-3))                   # set of horizontal obstacles
            obstacles3 = list(range(-2+4*cols,-2+(rows+1)*cols, cols))   # set of vertical obstacles near the end
            self.obstacles = obstacles1 + obstacles2 + obstacles3        # concatenate them all 

        # upward winds intensity for each column
        elif self.style=='windy':
            self.wind = [0,0,0,1,1,1,2,2,1,0] # as in example 6.5 of the book
    
    # override the step() function so that it can deal with wind
    def step(self, a, *args):
        a = self.actions[a]
        if not self.invalid(self.s,a): self.s += a
        
        if self.style=='windy':
            maxwind = self.wind[self.s%self.cols]
            for wind in range(maxwind, 0, -1): # we need to try apply all the wind or at least part of it
                if not self.invalid(self.s, wind*self.cols): self.s += wind*self.cols; break
        
        self.trace.append(self.s)
        reward, done = self.reward()       # must be done in this order for the cliff reset to work properly
        return self.s_(), reward, done, {}, {} # empty dict for compatibility

#===========================differen Grid world styles with visualisation=========================

'''
    Ok, dealing with the bare minimum grid without visualisation makes it difficult to observe 
    the behaviour of an agent. Now we add a useful set of visualisation routines to make this 
    possible. 

    This is will add an overhead so we try to minimise it by only initialising and calling when 
    a render() function is called. 

    We are moving from 1-d list of states to a 2-d set of coordinates. We use the modulus % 
    and the floor division operators to achieve this. Both are built-in operators and very efficient. 
    The function to_pos() converts a state into its corresponding position coordinates. 

    The render function does all the heavy lifting of visualising the environment and the agent's 
    current state s. It basically renders a 2-d grid as per the dimension of the grid along the side 
    with the agent and any obstacles (which block the agent's pathway) or cliff cells (which cause the 
    agent to reinitialise its position to state s0). We also call a placeholder function render_(), 
    which will be called to render further info. such as the states' representation in the grid, 
    as we will see next.
'''

class Grid(Grid):
    def __init__(self, pause=0, figsize=None, **kw):
        super().__init__(**kw)
        
        self.figsize = figsize # desired figure size  
        self.figsize0 = (12, 2) # default figure size
        self.fig = None        # figure handle, may have several subplots        
        self.ax0 = None        # Grid subplot handle
        
        self.pause = pause     # pause to slow animation
        self.arrows = None     # policy arrows (direction of action with max value)
        
        # assuming env is not dynamic, otherwise should be moved to render() near self.to_pos(self.s)
        self.start = self.to_pos(self.s0)         
        self.goal1 = self.to_pos(self.goals[0])
        self.goal2 = self.to_pos(self.goals[1])
        self.cmap = colors.ListedColormap(['w', 'darkgray'])

    # state representation function that converts 1-d list of state representation into a 2-d coordinates
    def to_pos(self, s):
        return [s%self.cols + 1, s//self.cols + 1]

    #------------------------------------------initialise------------------------------------------------- 
    def init_cells(self, cells): 
        Cells = np.zeros((self.rows+1, self.cols+1),  dtype=bool)
        Cells[0,0] = True # to populate for drawing 
        poses = self.to_pos(np.array(cells))
        Cells[poses[1], poses[0]] = True
        return Cells[1:,1:]
    
    #------------------------------------------render ✍️-------------------------------------------------
    # this function is to protect render() called twice for Gridi
    def render(self, **kw):
        self.render__(**kw)

    # We have placed most of the render overhead in the render() function to keep the rest efficient.
    # this function must not be called directly; instead, render() is to be called
    def render__(self, underhood='', pause=None, label='', subplot=131, large=False, 
               animate=True, image=False, saveimg=False,  **kw):
        
        if self.figsize is None:
            self.figsize = self.figsize0   # (12, 2)
            if   self.rows==1:             self.figsize = (15,.5) 
            elif underhood=='Q':           self.figsize = (12, 3)#(20, 10)
            elif underhood=='V' and large: self.figsize = (12, 3)#(35, 25)                        
        if image: self.figsize = (17, 3)   # changing the default figure size is disallowed for games

        if self.fig is None: self.fig = plt.figure(1)
        #if self.ax0 is None: self.ax0 = plt.subplot(subplot)
        plt.gcf().set_size_inches(self.figsize[0], self.figsize[1])
            
        #if   animate: self.ax0 = plt.subplot(subplot)
        #elif image:   plt.cla() 
        self.ax0 = plt.subplot(subplot)
        if image and not animate: plt.cla()
        
        
        # get hooks for self properties
        rows, cols = self.rows, self.cols
        pos, start, goal1, goal2 = self.to_pos(self.s), self.start, self.goal1, self.goal2
        
        pause = self.pause if pause is None else pause
        
        # a set of properties for the grid subplot
        
        prop = {'xticks': np.linspace(0, cols, cols+1),     'xticklabels':[],
                'yticks': np.linspace(0, rows, rows+1)+.01, 'yticklabels':[],
                'xlim':(0, cols), 'ylim':(0, rows), 'xlabel': label} # useful info
        self.ax0.update(prop)
        self.ax0.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        if self.style not in ['maze', 'cliff']: self.ax0.grid(True)

        # robot visuals :-)
        mrgn = .6
        eyes = ['˚-˚','ˇ-ˇ','ˆ-ˆ'][self.s%2 if not self.s in self.goals else 2]
        eyes, body = (eyes, 'ro') if underhood!='Q' else ('' , 'co')
        
        # plot goals and start state
        for (x,y), s in zip([goal1, goal2, start], ['G', 'G', 'S']):
            self.ax0.text(x-mrgn, y-mrgn, s, fontsize=11) #x-mrgn-(.6 if s=='S' else 0)
        
        # plot robot
        self.ax0.text(pos[0]-mrgn-.17, pos[1]-mrgn-.25, eyes, fontsize=9)
        self.ax0.plot(pos[0]-mrgn,     pos[1]-mrgn,     body, markersize=12) 
        #self.ax0.plot(pos, body, markersize=15) # this causes the body not be up to date in later lessons

        # to reduce overhead, pre-store coordinates in the grid only when rendering is needed
        if self.X is None: 
            self.X, self.Y = np.array(self.to_pos(np.arange(self.nS))) 
            self.Ox, self.Oy = np.arange(cols+1), np.arange(rows+1)

        # underhood obstacles and cliffs
        if self.style=='maze':  
            if self.Obstacles is None: self.Obstacles = self.init_cells(self.obstacles)
            self.ax0.pcolormesh(self.Ox, self.Oy, self.Obstacles, edgecolors='lightgray', cmap=self.cmap)
        
        if self.style=='cliff': 
            if self.Cliffs is None: self.Cliffs = self.init_cells(self.cliffs)
            self.ax0.pcolormesh(self.Ox, self.Oy, self.Cliffs, edgecolors='lightgray', cmap=self.cmap)

        # this means that the user wants to draw the policy arrows (actions)
        if 'V' in kw and underhood=='': underhood='V'
        if 'Q' in kw and underhood=='': underhood='maxQ'
        if 'π' in kw and underhood=='': underhood='π'
        
        # a placeholder function for extra rendering jobs
        underhoods = [p.strip() for p in underhood.split(',') if p.strip()]
        for underhood in underhoods:
            render_ = getattr(self, 'render_'+ underhood)(**kw)
        
        # windy style needs a bespoke rendering
        if self.style =='windy': self.render_windy()

        if image: self.render_image(saveimg=saveimg)
            
        # to animate, clear and plot the Grid
        if animate: clear_output(wait=True); plt.show(); time.sleep(pause)
        #else: plt.subplot(subplot)
    
    #-------------------------helper functions for rendering policies and value functions-----------
    def render_(self, **kw):
        pass # a placeholder for another drawing if needed
    
    def render_image(self, **kw):
        pass # a placeholder for capturing and saving the Grid as images
    
    # renders all states numbers' representation on the grid
    def render_states(self, **kw):
        X,Y  = self.X, self.Y
        for s in range(self.nS): 
            self.ax0.text(X[s]-.5,Y[s]-.95, s, fontsize=8, color='k')

    # renders all rewards on the grid
    def render_rewards(self, **kw):
        X,Y  = self.X, self.Y
        for s in range(self.nS): 
            self.ax0.text(X[s]-.5,Y[s]-.3, self.reward(s)[0], fontsize=10, color='r')


# =============================================================================================
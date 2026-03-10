from env.grid.base import *

# ================================visualising the policy within the grid=======================

'''
    Next we further add more rendering routines to enhance and enrich the Grid class. 
    Mainly, these rendering routines will be used to visualise the policy of an agent either 
    via  π or via the Q function. The arrows are used with the quiver function, which yields much 
    faster results than using the plt.text(x,y, '→') function.
'''



class Grid(Grid):
    def __init__(self, **kw):
        super().__init__(**kw)

    def init_arrows(self):       
        self._left,      self._right,   self._down,       self._up       = tuple(range(0,4))
        self._left_down, self._left_up, self._right_down, self._right_up = tuple(range(4,8))
        
        # works for quiver and pos, max action can potentially go upto 8! if we are dealing with a grid world
        self.arrows = np.zeros((self.nA,2), dtype=int)
        
        self.arrows[self._left ] =[-1, 0]  # '←'
        self.arrows[self._right] =[ 1, 0]  # '→'
        
        if self.nA>2:
            self.arrows[self._down ] =[ 0,-1]  # '↓'
            self.arrows[self._up   ] =[ 0, 1]  # '↑'

        if self.nA>4:
            self.arrows[self._left_down ]=[-1,-1]  # '↓←'
            self.arrows[self._left_up   ]=[-1, 1]  # '↑←'
            self.arrows[self._right_down]=[ 1,-1]  # '→↓'
            self.arrows[self._right_up  ]=[ 1, 1]  # '→↑'

    # returns all argmaxes, useful for when we have multiple maxes as np.argmax just returns the first 
    def argmaxeRows(self, Q): 
        return [np.where(Qs==Qs.max())[0] for Qs in Q]

    def render_maxQ(self, Q, **kw):
        self.render_π(π=Q.argmax(1))
        
    # renders a deterministic or stochastic policy based on Q or direct probability using π
    def render_π(self, π=None, **kw):
        π = np.ones((self.nS, self.nA )) if π is None else np.array(πTi[π]) # πTi[π] in case user passed '←' instead of indexes
        if self.arrows is None: self.init_arrows()
        X, Y = self.X, self.Y
       
        ind = np.array([s for s in range(self.nS) if not s in self.obstacles + self.goals + self.cliffs], dtype=np.uint32)
        if ind.any()==False: return
        
        if len(π.shape)==1: # deterministic policy
            U, Z = self.arrows[π].T
            plt.quiver(X[ind]-.5,Y[ind]-.5,  U[ind],Z[ind],color='b')
        elif len(π.shape)==2: # stochastic policy, shows multiple arrows if their probabilities are equal
            argmaxesπ = self.argmaxeRows(π[ind])
            for i in range(max(map(len, argmaxesπ))):
                argmaxes = [argmax[min(len(argmax)-1,i)] for argmax in argmaxesπ]
                U, Z = self.arrows[argmaxes].T
                plt.quiver(X[ind]-.5,Y[ind]-.5,  U,Z,color='b')

    # renders state value function
    def render_V(self, V=None, **kw):
        if V is None: V=np.ones(self.nS)
        X,Y  = self.X, self.Y
        fntsz, clr = 14 - int(self.cols/5), 'b'
        for s in range(self.nS):
            if s in self.obstacles or s in self.goals: continue
            plt.text(X[s]-.7,Y[s]-.7, '%.1f  '% V[s], fontsize=fntsz, color=clr) 
    
    # renders action-state value function
    def render_Q(self, Q=None, **kw):
        if Q is None: Q=np.ones((self.nS, self.nA ))
        X,Y  = self.X, self.Y
        fntsz, mrgn, clr = 12 - (5-self.nA) - int(self.cols/5), 0.4, 'b'
        for s in range(self.nS):
            if s in self.obstacles: continue        
            #  '→', '←', '↑', '↓'
            plt.text(X[s]-mrgn,Y[s]-mrgn, '←%.2f, '% Q[s,0], ha='right', va='bottom', fontsize=fntsz, color=clr) 
            plt.text(X[s]-mrgn,Y[s]-mrgn, '%.2f→  '% Q[s,1], ha='left' , va='bottom', fontsize=fntsz, color=clr)
            if self.nA==2: continue
            plt.text(X[s]-mrgn,Y[s]-mrgn, '↓%.2f, '% Q[s,2], ha='right', va='top'   , fontsize=fntsz, color=clr) 
            plt.text(X[s]-mrgn,Y[s]-mrgn, '%.2f↑  '% Q[s,3], ha='left' , va='top'   , fontsize=fntsz, color=clr) 



# ===================================== Visualisation for a specialist Grids =====================================

class Grid(Grid):
    def __init__(self, **kw):
        super().__init__(**kw)
        
        # randwalk related
        self.letters = None                    # letter rep. for states
        
    #--------------------------helper functions specific for some env and exercises---------
    # renders winds values on a grid
    def render_windy(self, **kw):
        for col in range(self.cols): # skipping the first and final states
            plt.text(col+.2,-.5, self.wind[col], fontsize=13, color='k')
        plt.text(6.15,1, '⬆',fontsize=60, color='lightgray')
        plt.text(6.15,4, '⬆',fontsize=60, color='lightgray')
    
    # renders a trace path on a grid
    def render_trace(self, **kw):
        poses = self.to_pos(np.array(self.trace))
        plt.plot(poses[0]-.5, poses[1]-.5, '->c')

    def render_V(self, **kw):
        super().render_V(**kw)
        if self.rows==1: self.render_letters()

    # renders all states letters' reprsentation on the grid
    def render_letters(self, **kw): # for drawing states numbers on the grid
        if self.nS>26: return
        X,Y  = self.X, self.Y
        # to reduce overhead, create the list only when render_letters is needed
        if self.letters is None: self.letters = self.letters_list() 
        for s in range(1,self.nS-1): # skipping the first and final states
            plt.text(X[s]-.5,Y[s]+.02, self.letters[s], fontsize=13, color='g')
    
    def letters_list(self, **kw):
        letters = [chr(letter) for letter in range(ord('A'),ord('A')+(self.nS-2))]
        letters.insert(0, 'G1')
        letters.append('G2')
        return letters

#==================================jumping grid !============================================
'''
    We can define a class that allows the agent to jump randomly or to a specific location 
    in the grid without going through intermediate states. This will be used later in other 
    lessons that deal with state representations. Here we pass jGrid to the maze function to 
    obtain an instance of a jumping Grid class without redefining the maze.
'''


class Grid(Grid):
    def __init__(self, jump=1, randjump=True, **kw):
        super().__init__(**kw)
        self.jump = jump
        self.randjump = randjump
        
    #-----------------------------------actions related---------------------------------------
    def step(self, a):
        jump = randint(1, min(self.jump, self.nS - self.s) +1) if self.randjump else self.jump
        if self.jump==1: return super().step(a)
            
        a = self.actions[a]*jump
        if not self.invalid(self.s, a):  
            #print('valid jump')
            self.s += a
        else: 
            #print('invalid jump')
            self.s = max(min(self.s+a, self.nS-1),0)
        
        self.trace.append(self.s)
        reward, done = self.reward() 
        return self.s_(), reward, done, {}, {}
    

# ======================A set of handy functions that will be used a lot===========================
class Mazes:
    def __init__(self, m=6, **kw):
        gridsizes = [(6, 9), (9 , 12), (12, 18), (16, 26), (24, 34), (32, 50), (39, 81), (60, 104)][:m]
        self.env = []
        for rows,cols in gridsizes:
            self.env.append(maze(r=rows,c=cols, **kw))
    
    def __getitem__(self, i): return self.env[i]
    
    def sizes(self):
        sizes = [0]
        for mz in  self.env: sizes.append(mz.nS_available())
        return sizes
        
#-------------------------------suitable for control------------------------------------------------
def grid(Grid=Grid, **kw):
    return Grid(gridsize=[8, 10], s0=31, goals=[36], **kw)

def grid8(Grid=Grid, **kw): 
    return grid(Grid=Grid, nA=8, **kw)

def windy(Grid=Grid,  **kw):
    return Grid(gridsize=[7, 10], s0=30, goals=[37], style='windy', **kw)

def cliffwalk(Grid=Grid, **kw):
    return Grid(gridsize=[4, 12], figsize=[12,1.5], s0=0,  goals=[11], style='cliff', reward='cliffwalk', **kw)

def maze(Grid=Grid, r=6, c=9, **kw):
    return Grid(gridsize=[r,c], s0=r//2*c, goals=[r*c-1], style='maze', **kw)

def maze_large(Grid=Grid, **kw):
    return maze(Grid=Grid, r=16, c=26, figsize=[25,4],**kw)

def maze8(Grid=Grid, **kw): 
    return maze(Grid=Grid, nA=8, **kw)


#-------------------------------suitable for prediction------------------------------------------------
def randwalk(Grid=Grid, nS=5+2, Vstar=None, **kw):
    if Vstar is None: Vstar = np.arange(0,nS)/(nS-1)
    return Grid(gridsize=(1,nS), reward='randwalk', nA=2, goals=[0,nS-1], s0=nS//2, Vstar=Vstar, **kw)

def randwalk_(Grid=Grid, nS=19+2, Vstar=None, **kw):
    if Vstar is None: Vstar = np.arange(-(nS-1),nS,2)/(nS-1)
    return Grid(gridsize=(1,nS), reward='randwalk_', nA=2, goals=[0,nS-1], s0=nS//2, Vstar=Vstar,**kw)

# ------------------------------change default size of cells in jupyter: an indicaiton of successful import----
def resize_cells(size=90):
    display(HTML('<style>.container {width:' +str(size) +'% !important}</style>'))
    

resize_cells()
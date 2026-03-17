class RoboEnvOdom(RobEnv):
    def __init__(self, sectors=12, **kw):
        super().__init__(**kw)
        self.k = sectors
        self.nF = sectors + 2     # sectors + goal dist + goal angle
        print('state size =', self.nF)

    # reward encouraging progress and safe navigation
    def reward_(self, a):
        reward = sum([
            -0.5,                    # step penalty
            2*self.Δgoal_dist,       # progress reward
            .5*self.Δθgoal_dist,     # heading improvement
            -4*self.at_wall,         # collision penalty
            10*self.at_goal          # goal reward
        ])
        if self.verbose and reward > -1: print('reward =', reward)
        return reward
    
    def s_(self):
        φ    = self.φnearest(k=self.k)
        d, θ = self.nearest()

        # normalise all
        d /= hypot(self.xdim, self.ydim)
        θ /= pi
        φ -= self.min_range
        φ /= self.max_range - self.min_range
        
        # return normalised sectors, distance and angular distance
        return np.r_[φ, d, θ].astype(np.float32)


# ====================================================================================================
class RoboEnvScan(RobEnv):

    def __init__(self, sectors=16, **kw):
        super().__init__(**kw)
        self.k = sectors
        self.nF = sectors
        print('state size =', self.nF)

    def reward_(self, a):
        reward = sum([
            -1,
            1.5*self.Δgoal_dist,
            -5*self.at_wall,
            8*self.at_goal
        ])

        if self.verbose and reward>-1: print('reward =', reward)
        return reward

    def s_(self):
        φ = self.φnearest(k=self.k)
        φ -= self.min_range
        φ /= self.max_range - self.min_range
        return (φ <.35).astype(np.float32)
        
import numpy as np
    
def gap_filling(position, visibility):
    duration        = np.shape(position)[1]
    new_position    = np.zeros((3,duration))
    new_visibility  = np.ones((duration))
    new_visibility[visibility==0] = 0
    new_position   += position
    
    if np.sum(visibility) == 0:
        print('invisible')
        return position
        
    else:
        ## Beginning
        if visibility[0] == 0:
            t = np.argmax(visibility)
            new_position[:,:t] = position[:,t].repeat(t).reshape(3,t)
            new_visibility[:t] = np.ones(t)

        ## End
        if visibility[-1] == 0:
            reversed = new_visibility[::-1]
            t = np.argmax(reversed)
            new_position[:,-t:] = position[:,-t-1].repeat(t).reshape(3,t)
            new_visibility[-t:] = np.ones(t)	


        while np.mean(new_visibility) < 1:
            # Find the first hole
            tmin = np.argmin(new_visibility)
            tmax = tmin + np.argmax(new_visibility[tmin:])
            # Fill it in
            new_position[:,tmin:tmax] = np.array([position[:,tmin-1]]).T + np.array([position[:,tmax] - position[:,tmin-1]]).T*np.array([np.arange(1,tmax - tmin + 1)])/(tmax - tmin + 1.)
            new_visibility[tmin:tmax] = np.ones(tmax-tmin)
        return new_position
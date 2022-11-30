import numpy as np
from LocalFrameTransform import minimum_image
# Op ... H-O
def ishbond(Op,O,H,lbox):
    if np.linalg.norm(minimum_image(Op-O,lbox)) < 3.5:
        v1 = minimum_image(H-O,lbox)
        v2 = minimum_image(Op-O,lbox)
        if np.inner(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2) > np.cos(30*np.pi/180):
            return True
    return False
import xasy2asy as x2a
import numpy as np
import PyQt5.QtCore as Qc


class PrimitiveShape:
    # The magic number.
    # see https://www.desmos.com/calculator/lw6j7khikj for unitcircle
    # optimal_ctl_pt = 0.5447

    @classmethod
    def circle(cls, position, radius):
        if isinstance(position, tuple) or isinstance(position, np.ndarray):
            pos_x, pos_y = position
        elif isinstance(position, Qc.QPoint) or isinstance(position, Qc.QPointF):
            pos_x = position.x()
            pos_y = position.y()
        else:
            raise TypeError("Position must be a valid type!")

        newCircle = x2a.asyPath()
        ptsList = [(pos_x + radius, pos_y), (pos_x, pos_y + radius), (pos_x - radius, pos_y), (pos_x, pos_y - radius),
                   (pos_x + radius, pos_y)]
        # cycle doesn't work for now.
        lkList = ['..', '..', '..', '..']
        newCircle.initFromNodeList(ptsList, lkList)
        return newCircle

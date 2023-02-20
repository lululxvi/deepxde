import deepxde as dde
import numpy as np
## rectangle
# https://stackoverflow.com/a/43082119/14598633
#geom = dde.geometry.Rectangle(np.array([[0,0,0,0], [0,0,0,0],[0,0,0,0]]),np.array([[2,3,1,1],[2,3,1,1],[2,3,1,1]]))

#geom = dde.geometry.Rectangle(np.array([0,0]),np.array([2,3]))
#geom = dde.geometry.Rectangle(np.array([0,0]),[2,3])
#geom = dde.geometry.Rectangle([0,0],np.array([2,3]))

#geom = dde.geometry.Rectangle([0,0],[2,3])
#geom = dde.geometry.Rectangle([0,0,0],[2,3,1])

# cuboid
#geom = dde.geometry.Cuboid(np.array([[0,0,0,0], [0,0,0,0],[0,0,0,0]]),np.array([[2,3,1,1],[2,3,1,1],[2,3,1,1]]))

#geom = dde.geometry.Cuboid(np.array([0,0]),np.array([2,3]))
#geom = dde.geometry.Cuboid(np.array([0,0]),[2,3])
#geom = dde.geometry.Cuboid([0,0],np.array([2,3]))

#geom = dde.geometry.Cuboid([0,0],[2,3])
#geom = dde.geometry.Cuboid([0,0,0],[2,3,1])

# sphere
#geom = dde.geometry.Sphere(3,4.2)
#geom = dde.geometry.Sphere([3,2],[4.2])
#geom = dde.geometry.Sphere([3,2],4.2)
#geom = dde.geometry.Sphere([3,2,1],4.2)

# disk

#geom = dde.geometry.Disk(3,4.2)
#geom = dde.geometry.Disk([3,2],[4.2])
#geom = dde.geometry.Disk([3,2],4.2)
#geom = dde.geometry.Disk([3,2],4)
#geom = dde.geometry.Disk([3,2,1],4)

# triangle
#geom = dde.geometry.Triangle(np.array([0,0,0]),[1,1,0],[0,2,0])
#geom = dde.geometry.Triangle([0,0,0],[1,1,0],[0,2,0])

#geom = dde.geometry.Triangle([0,0],[1,1],[0,2])

#geom = dde.geometry.Triangle([0,0],[1,1],[2,2])

#print(np.shape(np.array(np.array([1,2]))))
from dolfin import *
from mshr import *

def Create_Mesh(L, H, indent, R, count, res):
	LL = L + indent*2
	domain = Rectangle(Point(0.0, 0.0), Point(LL, H))
	up_down = True
	x = indent+R
	y = int(up_down)
	h = float(L/count)
	for i in range(count):
		circ = Circle(Point(x, y), R, 32)
		domain = domain - circ
		x += h
		up_down = not up_down
		y = int(up_down)
	mesh = generate_mesh(domain, float(res))
	info(mesh)
	return mesh

def Mark_Boundaries(meshname, L, H, indent):
	L += indent*2
	# Define boundaries
	class Top(SubDomain):
	    def inside(self, x, on_boundary):
		    return between(x[0], (0.0, L)) and between(x[1], (H-DOLFIN_EPS*100000, H+DOLFIN_EPS*100000)) and on_boundary
		    
	class Bottom(SubDomain):
	    def inside(self, x, on_boundary):
		    return between(x[0], (0.0, L)) and between(x[1], (-DOLFIN_EPS*1000, DOLFIN_EPS*1000)) and on_boundary

	class Left(SubDomain):
	    def inside(self, x, on_boundary):
		    return near(x[0], 0.0) and on_boundary

	class Right(SubDomain):
	    def inside(self, x, on_boundary):
		    return near(x[0], L) and on_boundary

	class Obstacle(SubDomain):
	    def inside(self, x, on_boundary):
	        return on_boundary

	mesh = Mesh(meshname)
	boundaries = MeshFunction("size_t", mesh, dim=1)
	boundaries.set_all(0)
	Obstacle().mark(boundaries, 3)
	Left().mark(boundaries, 1)
	Right().mark(boundaries, 2)
	Top().mark(boundaries, 4)
	Bottom().mark(boundaries, 4)
	return boundaries


L = 10.0      # active domain lenght
H = 1.0       # domain height
indent = 1.2  # free space before and after obstacles
R = 0.4       # obstacle radius
count = 11    # obstacle count
res = 200     # grid resolution

mesh = Create_Mesh(L, H, indent, R, count, res)
File("./res/mesh.pvd") << mesh
File("mesh.xml") << mesh

boundaries = Mark_Boundaries(mesh, L, H, indent)
File("./res/boundaries.pvd") << boundaries
File("boundaries.xml") << boundaries

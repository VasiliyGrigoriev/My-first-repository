from dolfin import *
import numpy as np

def Model(Pe_, Daa_, Dad_):
	set_log_level(35)
	Pe = Pe_#Constant('0.4')
	Daa = Daa_#Constant('0.0055')
	Dad = Dad_#Constant('0.045')

	# Mesh
	mesh = Mesh("mesh.xml")
	boundaries = MeshFunction("size_t", mesh, "boundaries.xml")
	
	ds = Measure('ds')(subdomain_data=boundaries)
	n = FacetNormal(mesh)

	# Function space for the PDE solution
	V = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
	VV = V * V
	W = FunctionSpace(mesh, VV)

	# Define unknown and test function(s)
	(v, vm) = TestFunctions(W)
	w = Function(W)
	(u, m) = (w[0], w[1])
	# Solution at the previous time level
	w_old = Function(W)
	(u_old, m_old) = (w_old[0], w_old[1])

	# Velocity function
	Q = VectorFunctionSpace(mesh, "CG", 2)
	vel = Function(Q, "vel.xml")

	#plot(vel, interactive=True)

	nx = Expression(('1.0','0.0'), degree=1, domain = mesh)
	psi = dot(vel,nx)


	# Define boundary conditions
	bcs = DirichletBC(W.sub(0), Constant(1.0),  boundaries, 1)

	# Initial condition
	w0 = interpolate( Constant((0., 0.)), W)

	# Set the options for the time discretization
	T = 40.0
	t = 0.0
	Nt = 400
	step = T / Nt

	# Define the variational formulation of the problem
	F = (u - u_old)*v*dx() - 0.5*step*(u+u_old)*inner(vel, grad(v))*dx() \
		       + 0.5*(step/Pe)*inner(grad(u+u_old), grad(v))*dx() \
		       + 0.5*step*(u+u_old)*psi*v*ds(2) \
		       + (m - m_old)*v*ds(3)
	F = F + (m - m_old)*vm*dx() + 0.5*step*Dad*(m+m_old)*vm*dx() \
		       - 0.5*step*Daa*(u+u_old)*vm*dx()

	td = np.zeros((Nt), 'float')
	cd = np.zeros((Nt), 'float')

	# Execute the time loop

	f1 = File("./res/c.pvd")
	f2 = File("./res/m.pvd")


	w_old.assign(w0)
	for i in range(Nt):
		print(t, T)
		t += step

		solve(F == 0, w, bcs)
		w_old.assign(w)

		(u, m) = w.split()
		f1<<u
		f2<<m

		ii = u * ds(2)
		I = assemble(ii)
		td[i] = t
		cd[i] = I
		
	return td, cd

if __name__=='__main__':
	td, cd = Model(0.05, 0.005, 0.05)
	np.save("td.npy", td)
	np.save("cd.npy", cd)

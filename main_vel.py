from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
time1 = time.time()

# Mesh loading
mesh = Mesh("mesh.xml")
boundaries = MeshFunction("size_t", mesh, "boundaries.xml")

ds = Measure('ds')(subdomain_data=boundaries)
n = FacetNormal(mesh)

# Function spaces for velocity
P2 = VectorElement('Lagrange', mesh.ufl_cell(), 2)
P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

V = FunctionSpace(mesh, P2)

# Define unknown and test function(s)
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

# Define boundary conditions
zero_vec = Expression(("0.0", "0.0"), degree=1)

bcs = [DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 1),\
       DirichletBC(W.sub(0).sub(0), Constant(1.0), boundaries, 1),\
       DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 4),\
       DirichletBC(W.sub(0), zero_vec, boundaries, 3),\
       DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)]       

# Form definition
Re = 1.0
g = Constant(1.0)

F = inner(grad(u)*u, v)*dx + 1.0/Re*inner(grad(u), grad(v))*dx + div(u)*q*dx - div(v)*p*dx 

# Compute solution
solve(F == 0, w, bcs, solver_parameters={"newton_solver":
                      {"relative_tolerance": 1e-8, "relaxation_parameter": 1.0}})
(u, p) = w.split()
plot(u, title = "Velocity")
plot(p, title = "Pressure")
File("./res/u.pvd")<<u
File("./res/p.pvd")<<p

i = Function(V)

assigner = FunctionAssigner(V, W.sub(0))
assigner.assign(i, u)
File("./vel.xml")<<i

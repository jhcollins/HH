"""
Artificial Reef Flume
"""
import numpy as np
from math import *#sqrt
from proteus import (Domain, Context,
                     FemTools as ft,
                     #SpatialTools as st,
#                     MeshTools as MeshTools,
                     WaveTools as wt)
from proteus.mprans import SpatialTools as st
from proteus.Profiling import logEvent
from proteus.mprans.SpatialTools import Tank2D
from proteus.ctransportCoefficients import smoothedHeaviside, smoothedHeaviside_integral
from proteus.Gauges import PointGauges, LineIntegralGauges, LineGauges

# from math import *
import proteus.MeshTools
#import sys
#sys.path.insert(0, 'geom')
import ss_geom as ssg

shrink=410

# predefined options
opts=Context.Options([
    # water column 
    ("water_level", 0.6,"Height of water column in m"),
    ("tailwater", 0.6,"Height of water column in m"),
    ("water_width",304, "width (along x) of  water column in m"),
    # tank
    ("tank_dim", (3.22, 1.8), "Dimensions of the tank  in m"),
    #gravity 
    ("g",(0,-9.81,0), "Gravity vector in m/s^2"),
    # gauges
    ("gauge_output", True, "Produce gauge data"),
    ("gauge_location_p", (3.22, 0.12, 0), "Pressure gauge location in m"),
    # mesh refinement and timestep
    ("refinement", 1 ,"Refinement level, he = L/(4*refinement - 1), where L is the horizontal dimension"), 
    ("he",0.1, "element diameter"),
    ("cfl", 0.1 ,"Target cfl"),
    # run time options
    ("T", 0.1 ,"Simulation time in s"),
    ("dt_fixed", 0.01, "Fixed time step in s"),
    ("dt_init", 0.001 ,"Maximum initial time step in s"),
    ("useHex", False, "Use a hexahedral structured mesh"),
    ("structured", False, "Use a structured triangular mesh"),
    ("waveheight", 1.0, "wave height"),
    ("wavelength", 10.0, "wave length"),
    ("wallheight",5.0, "levee height"),
    ("HtoV",3.0,"levee horizontal to vertical slope H:V"),
    ("leeward_dry",True, "no water on leeward side"),
    ("gen_mesh", True ,"Generate new mesh"),
    ("test",True, "debugging"),
    ("upstream_length", None, "distance from inflow boundary to weir"),
    ("downstream_length", None, "distance from plunge pool to outflow boundary"),
    ("top", 1.0, "height of the top boundary"),
    ("flowrate",1,"unit flowrate for 2d (assuming thickness of 1m)"),
    ("kInflow",0.0,"kappa at the inflow boundary"),
    ("xs",1.0, "x position of reef feature from outflow"),
    ("x1",0.5," x distance from xs of the peak"),
    ("y1",0.5," y distance from 0.0 of the peak"),
    ("xe",2.0," x extent of reef from xs"),
    ("Len",5.0," length of flume"),
    ("useRANS",0,"turbulence model")
    ])


kInflow=opts.kInflow
he=opts.he
# ----- CONTEXT ------ #

# water
waterLine_z = opts.water_level
waterLine_x = opts.water_width
tailwater=opts.tailwater

#u=opts.flowrate/opts.water_level

##########################################
#     Discretization Input Options       #
##########################################

#[temp] temporary location
backgroundDiffusionFactor = 0.01

refinement = opts.refinement
genMesh = opts.gen_mesh
movingDomain = False
checkMass = False
applyRedistancing = True
useOldPETSc = False
useSuperlu = False
timeDiscretization = 'be'  # 'vbdf', 'be', 'flcbdf'
spaceOrder = 1
useHex = opts.useHex
structured = opts.structured
useRBLES = 0.0
useMetrics = 1.0
applyCorrection = True
useVF = 1.0
useOnlyVF = False
useRANS = opts.useRANS  # 0 -- None
             # 1 -- K-Epsilon
             # 2 -- K-Omega

# ----- INPUT CHECKS ----- #
if spaceOrder not in [1,2]:
    raise ValueError("INVALID: spaceOrder(" + str(spaceOrder) + ")")

if useRBLES not in [0.0, 1.0]:
    raise ValueError("INVALID: useRBLES(" + str(useRBLES) + ")")

if useMetrics not in [0.0, 1.0]:
    raise ValueError("INVALID: useMetrics(" + str(useMetrics) + ")")

# ----- DISCRETIZATION ----- #
nd = 2
if spaceOrder == 1:
    hFactor = 1.0
    if useHex:
        basis = ft.C0_AffineLinearOnCubeWithNodalBasis
        elementQuadrature = ft.CubeGaussQuadrature(nd, 2)
        elementBoundaryQuadrature = ft.CubeGaussQuadrature(nd - 1, 2)
    else:
        basis = ft.C0_AffineLinearOnSimplexWithNodalBasis
        elementQuadrature = ft.SimplexGaussQuadrature(nd, 3)
        elementBoundaryQuadrature = ft.SimplexGaussQuadrature(nd - 1, 3)
elif spaceOrder == 2:
    hFactor = 0.5
    if useHex:
        basis = ft.C0_AffineLagrangeOnCubeWithNodalBasis
        elementQuadrature = ft.CubeGaussQuadrature(nd, 4)
        elementBoundaryQuadrature = ft.CubeGaussQuadrature(nd - 1, 4)
    else:
        basis = ft.C0_AffineQuadraticOnSimplexWithNodalBasis
        elementQuadrature = ft.SimplexGaussQuadrature(nd, 4)
        elementBoundaryQuadrature = ft.SimplexGaussQuadrature(nd - 1, 4)

##########################################
# Numerical Options and Other Parameters             #
##########################################

weak_bc_penalty_constant = 10.0
nLevels = 1

# ----- PHYSICAL PROPERTIES ----- #

# Water
rho_0 = 998.2
nu_0 = 1.004e-6

# Air
rho_1 = 1.205
nu_1 = 1.500e-5

# Surface Tension
sigma_01 = 0.0

# Gravity
g = opts.g

# ----- TIME STEPPING & VELOCITY----- #

T = opts.T
dt_fixed = opts.dt_fixed
dt_init = min(0.1 * dt_fixed, opts.dt_init)
runCFL = opts.cfl
nDTout = int(round(T / dt_fixed))

# ----- DOMAIN ----- #

nLevels = 1
#parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.element
parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0


#--------------------------------------------------------------------------------------------#
boundaries=['inflow','outflow','bottom','top']
boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])
print boundaryTags

vertices=[[0.0, 0.0],#0
          [opts.xs, 0.0],
          [opts.x1+opts.xs, opts.y1],
          [opts.xe+opts.xs, 0.0],
          [opts.Len, 0.0],
          [opts.Len, opts.top],
          [0.0, opts.top]]


vertexFlags=[boundaryTags['outflow'],
             boundaryTags['bottom'],
             boundaryTags['bottom'],
             boundaryTags['bottom'],
             boundaryTags['bottom'],
             boundaryTags['inflow'],
             boundaryTags['top']]

segments=[[0,1],
          [1,2],
          [2,3],
          [3,4],
          [4,5],
          [5,6],
          [6,0]]

segmentFlags=[boundaryTags['bottom'],
              boundaryTags['bottom'],
              boundaryTags['bottom'],
              boundaryTags['bottom'],
              boundaryTags['inflow'],
              boundaryTags['top'],
              boundaryTags['outflow']]


regx=(vertices[1][0]+vertices[0][0])/2
regy=(vertices[-1][1]+vertices[0][1])/2
regions=[[regx, regy]]
regionFlags=[1]

 
domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                              vertexFlags=vertexFlags,
                                              segments=segments,
                                              segmentFlags=segmentFlags,
                                              regions = regions,
                                              regionFlags = regionFlags,
                                              )

he = opts.he

domain.MeshOptions.setParallelPartitioningType('node')
domain.boundaryTags = boundaryTags
#domain.readPoly("mesh")
domain.writePoly("mesh")
domain.writePLY("mesh")
domain.writeAsymptote("mesh")
#triangleOptions = "VApq30Dena%8.8f"# % ((he**2)/2.0,)
triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)
logEvent("""Mesh generated using: tetgen -%s %s"""  % (triangleOptions,domain.polyfile+".poly"))
domain.MeshOptions.triangleOptions=triangleOptions
#--------------------------------------------------------------------------------------------#  






waterLevel= opts.water_level
bottom=min(vertices, key=lambda x: x[1])[1]
topy=max(vertices, key=lambda x: x[1])[1]

# ----- STRONG DIRICHLET ----- #

ns_forceStrongDirichlet = False

# ----- NUMERICAL PARAMETERS ----- #

if useMetrics:
    ns_shockCapturingFactor = 0.25
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor = 0.25
    ls_lag_shockCapturing = True
    ls_sc_uref = 1.0
    ls_sc_beta = 1.0
    vof_shockCapturingFactor = 0.25
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    rd_shockCapturingFactor = 0.25
    rd_lag_shockCapturing = False
    epsFact_density = epsFact_viscosity = epsFact_curvature \
                    = epsFact_vof = ecH \
                    = epsFact_consrv_dirac = epsFact_density \
                    = 3.0
    epsFact_consrv_heaviside = 3.0
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 0.1
    redist_Newton = True
    kappa_shockCapturingFactor = 0.25
    kappa_lag_shockCapturing = True  #False
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.0
    dissipation_shockCapturingFactor = 0.25
    dissipation_lag_shockCapturing = True  #False
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.0
else:
    ns_shockCapturingFactor = 0.9
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor = 0.9
    ls_lag_shockCapturing = True
    ls_sc_uref = 1.0
    ls_sc_beta = 1.0
    vof_shockCapturingFactor = 0.9
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    rd_shockCapturingFactor = 0.9
    rd_lag_shockCapturing = False
    epsFact_density = epsFact_viscosity = epsFact_curvature \
        = epsFact_vof = ecH \
        = epsFact_consrv_dirac = epsFact_density \
        = 1.5
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 1.0
    redist_Newton = False
    kappa_shockCapturingFactor = 0.9
    kappa_lag_shockCapturing = True  #False
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.0
    dissipation_shockCapturingFactor = 0.9
    dissipation_lag_shockCapturing = True  #False
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.0

# ----- NUMERICS: TOLERANCES ----- #

ns_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
vof_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
ls_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
rd_nl_atol_res = max(1.0e-10, 0.005 * he)
mcorr_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
kappa_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
dissipation_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)

# ----- TURBULENCE MODELS ----- #

ns_closure = 0  #1-classic smagorinsky, 2-dynamic smagorinsky, 3 -- k-epsilon, 4 -- k-omega
if useRANS == 1:
    ns_closure = 3
elif useRANS == 2:
    ns_closure = 4

##########################################
#  Initial conditions for free-surface   #
##########################################

def signedDistance(X):

    return X[1]-opts.water_level

#--------------------------------------------------------------------------------------------#  


def twpflowVelocity_u(X,t):
    waterspeed = u
    H = smoothedHeaviside(epsFact_consrv_heaviside*he,X[1]-waterLevel)
    if t<1:
        return (1.0-H)*waterspeed*t
    else:
        return (1.0-H)*waterspeed

u=opts.flowrate/(opts.water_level-vertices[0][1])

def twpflowVelocity_u_flux(X,t):
    waterspeed = u
    H = smoothedHeaviside(epsFact_consrv_heaviside*he,X[1]-waterLevel)
    if t<1:
	return -1*(1.0-H)*waterspeed*t
    else:
	return -1*(1.0-H)*waterspeed

def intwpflowPressure(X,t):
    p_top = 0.0
    phi_top = topy - waterLevel
    phi = X[1] - waterLevel
    return p_top - g[1]*(rho_0*(phi_top - phi) + \
                         (rho_1 -rho_0) * \
                         (smoothedHeaviside_integral(epsFact_consrv_heaviside*he,phi_top)
                          -
                          smoothedHeaviside_integral(epsFact_consrv_heaviside*he,phi)))

def outtwpflowPressure(X,t):
    p_top = 0.0
    phi_top = topy - tailwater
    phi = X[1] - tailwater
    return p_top - g[1]*(rho_0*(phi_top - phi) + \
                         (rho_1 -rho_0) * \
                         (smoothedHeaviside_integral(epsFact_consrv_heaviside*he,phi_top)
                          -
                          smoothedHeaviside_integral(epsFact_consrv_heaviside*he,phi)))

def inbcVF(X,t):
    return smoothedHeaviside(epsFact_consrv_heaviside*he, X[1] - waterLevel)

def inbcPhi(X,t):
    return X[1] - waterLevel

def outbcVF(X,t):
    return smoothedHeaviside(epsFact_consrv_heaviside*he, X[1] - tailwater)

def outbcPhi(X,t):
    return X[1] - tailwater

from proteus.default_p import *
from proteus.mprans import RANS2P
import numpy as np
from proteus import Context
from ar import *
ct = Context.get()
domain = ct.domain
nd = domain.nd
triangleOptions = ct.triangleOptions
print triangleOptions
mesh = domain.MeshOptions.triangleOptions

#genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T  # might not be necessaryd

LevelModelType = RANS2P.LevelModel
if ct.useOnlyVF:
    LS_model = None
else:
    LS_model = 2
if ct.useRANS >= 1:
    Closure_0_model = 5
    Closure_1_model = 6
    if ct.useOnlyVF:
        Closure_0_model = 2
        Closure_1_model = 3
    if ct.movingDomain:
        Closure_0_model += 1
        Closure_1_model += 1
else:
    Closure_0_model = None
    Closure_1_model = None

coefficients = RANS2P.Coefficients(epsFact=ct.epsFact_viscosity,
                                   sigma=0.0,
                                   rho_0=ct.rho_0,
                                   nu_0=ct.nu_0,
                                   rho_1=ct.rho_1,
                                   nu_1=ct.nu_1,
                                   g=ct.g,
                                   nd=nd,
                                   ME_model=int(ct.movingDomain)+0,
                                   VF_model=int(ct.movingDomain)+1,
                                   LS_model=int(ct.movingDomain)+LS_model,
                                   Closure_0_model=Closure_0_model,
                                   Closure_1_model=Closure_1_model,
                                   epsFact_density=ct.epsFact_density,
                                   stokes=False,
                                   useVF=ct.useVF,
                                   useRBLES=ct.useRBLES,
                                   useMetrics=ct.useMetrics,
                                   eb_adjoint_sigma=1.0,
                                   eb_penalty_constant=ct.weak_bc_penalty_constant,
                                   forceStrongDirichlet=ct.ns_forceStrongDirichlet,
                                   turbulenceClosureModel=ct.ns_closure,
                                   movingDomain=ct.movingDomain)
 
testing=1

if testing == 1:

    def getDBC_p(x,flag):
        if flag == boundaryTags['top']:
            return lambda x,t: 0.0
#        if flag == boundaryTags['inflow']:
#            return ct.intwpflowPressure
        if flag == boundaryTags['outflow']:
            return ct.outtwpflowPressure

    def getDBC_u(x,flag):
        if flag == boundaryTags['inflow']:
            return ct.twpflowVelocity_u
        if flag != boundaryTags['outflow']:
            return lambda x,t: 0.0

#        if flag == boundaryTags['left']:# and x[1]<waterLevel:
#            return u_wave1
#       return lambda x,t: 1.0
#        else:
#            return lambda x,t: 0.0

    def getDBC_v(x,flag):
#        return None#lambda x,t: 0.0
        if flag != boundaryTags['top']:
            return lambda x,t: 0.0
#        return lambda x,t: 0.0
#        if flag == boundaryTags['left']:
#            return w_wave1
        
    def getDBC_w(x,flag):
        return lambda x,t: 0.0
    
    # ----- ADVECTIVE FLUX BOUNDARY CONDITIONS ----- #
    def getAFBC_p(x,flag):
        if flag != boundaryTags['top']:
#            if flag != boundaryTags['inflow']:
            if flag != boundaryTags['outflow']:
                if flag == boundaryTags['inflow']:
                    return ct.twpflowVelocity_u_flux
                else:
                    return lambda x,t: 0.0

#            if flag != boundaryTags['right']:
#                if flag == boundaryTags['left']: #and x[1]<waterLevel:
#                    return u_wavef1
                #                return lambda x,t: -1.0
#                else:
#                    return lambda x,t: 0.0
        #return 0.0

    def getAFBC_u(x,flag):
#        if flag != boundaryTags['left']: 
#        return lambda x,t: 0.0
        return None
#        else:
#            return lambda x,t: 0.0
    def getAFBC_v(x,flag):
        return None
#        if flag !=boundaryTags['top']:
#            return lambda x,t: 0.0  

    def getAFBC_w(x,flag):
        return None#return 0.0
    
      # ----- DIFFUSIVE FLUX BOUNDARY CONDITIONS ----- #
    def getDFBC_u(x,flag):
#        if flag != boundaryTags['left']:
#            return lambda x,t: 0.0
                    
        return lambda x,t: 0.0
    
    def getDFBC_v(x,flag):
#        if flag != boundaryTags['top']:
#            return lambda x,t: 0.0
                    
        return lambda x,t: 0.0
    
    def getDFBC_w(x,flag):
        return lambda x,t: 0.0
    
    class AtRest:
        def __init__(self):
            pass
        def uOfXT(self,x,t):
            return 0.0
                                                                                                                
        
# BOUNDARY CONDITIONS #
dirichletConditions = {0:getDBC_p,
                       1:getDBC_u,
                       2:getDBC_v}
advectiveFluxBoundaryConditions =  {0:getAFBC_p,
                                    1:getAFBC_u,
                                    2:getAFBC_v}
diffusiveFluxBoundaryConditions = {0:{},
                                   1:{1:getDFBC_u},
                                   2:{2:getDFBC_v}}
# INITIAL CONDITIONS #
initialConditions = {0:AtRest(),
                     1:AtRest(),
                     2:AtRest()}

#print topy
class PerturbedSurface_p:
    def __init__(self,waterLevel):
        self.waterLevel=waterLevel
    def uOfXT(self,x,t):
        if ct.signedDistance(x) < 0:
            return -(topy - self.waterLevel)*ct.rho_1*ct.g[1] - (self.waterLevel - x[1])*ct.rho_0*ct.g[1]
        else:
            return -(topy - self.waterLevel)*ct.rho_1*ct.g[1]

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0
#print ct.waterLine_z
initialConditions = {0:PerturbedSurface_p(ct.waterLine_z),
                     1:AtRest(),
                     2:AtRest()}

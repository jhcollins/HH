from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.mprans import VOF
from proteus import Context
from ss import *
ct = Context.get()
domain = ct.domain
nd = domain.nd
mesh = domain.MeshOptions


genMesh = mesh.genMesh
movingDomain = ct.movingDomain
T = ct.T

LevelModelType = VOF.LevelModel
if ct.useOnlyVF:
    RD_model = None
    LS_model = None
else:
    RD_model = 3
    LS_model = 2

coefficients = VOF.Coefficients(LS_model=int(ct.movingDomain)+LS_model,
                                V_model=int(ct.movingDomain)+0,
                                RD_model=int(ct.movingDomain)+RD_model,
                                ME_model=int(ct.movingDomain)+1,
                                checkMass=True,
                                useMetrics=ct.useMetrics,
                                epsFact=ct.epsFact_vof,
                                sc_uref=ct.vof_sc_uref,
                                sc_beta=ct.vof_sc_beta,
                                movingDomain=ct.movingDomain)

def getDBC_vof(x,flag):
    if flag ==boundaryTags['top']:
        return lambda x,t: 1.0
    elif flag == boundaryTags['inflow']:
        return ct.inbcVF
    elif flag == boundaryTags['outflow']:
        return ct.outbcVF
    else:
        return lambda x,t: 1.0
dirichletConditions = {0:getDBC_vof}
    
def getAFBC_vof(x,flag):
    if flag == boundaryTags['top']:
        return None
    elif flag == boundaryTags['inflow']:
        return None
    elif flag == boundaryTags['outflow']:
        return None
    else:
        return lambda x,t: 0.0
           
        
advectiveFluxBoundaryConditions = {0:getAFBC_vof}
diffusiveFluxBoundaryConditions = {0:{}}

class PerturbedSurface_H:
    def uOfXT(self,x,t):
        return smoothedHeaviside(epsFact_consrv_heaviside*he,signedDistance(x))
    
initialConditions  = {0:PerturbedSurface_H()}
    

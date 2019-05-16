from proteus import *
from dissipation_p import *

timeIntegration = TimeIntegration.BackwardEuler_cfl
stepController  = StepControl.Min_dt_cfl_controller

femSpaces = {0:basis}

massLumping       = False
numericalFluxType = Dissipation.NumericalFlux
conservativeFlux  = None
subgridError      = Dissipation.SubgridError(coefficients=coefficients,nd=nd)
shockCapturing    = Dissipation.ShockCapturing(coefficients,nd,shockCapturingFactor=dissipation_shockCapturingFactor,
                                         lag=dissipation_lag_shockCapturing)

fullNewtonFlag  = True
multilevelNonlinearSolver = Newton
levelNonlinearSolver      = Newton

nonlinearSmoother = None
linearSmoother    = None
#printNonlinearSolverInfo = True
matrix = SparseMatrix
if not useSuperlu:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver      = KSP_petsc4py
else:
    multilevelLinearSolver = LU
    levelLinearSolver      = LU

linear_solver_options_prefix = 'dissipation_'
levelNonlinearSolverConvergenceTest = 'rits'#'r'
linearSolverConvergenceTest         = 'rits'#'r'

tolFac = 0.0
linTolFac =0.0001
l_atol_res = 0.0001*dissipation_nl_atol_res
nl_atol_res = dissipation_nl_atol_res
useEisenstatWalker = False

maxNonlinearIts = 50
maxLineSearches = 0


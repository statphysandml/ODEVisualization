from odesolver.flow_equations import FlowEquations

# Import the main odevisualizationlib.flow to ensure base types are available
# This must be imported before importing lorentzattractorsimulation
import odevisualizationlib.flow


class LorentzAttractor(FlowEquations):
    def __init__(self):
        from lorentzattractorsimulation import LorentzAttractorFlow, LorentzAttractorJacobians
        super().__init__(flow=LorentzAttractorFlow(), jacobians=LorentzAttractorJacobians())

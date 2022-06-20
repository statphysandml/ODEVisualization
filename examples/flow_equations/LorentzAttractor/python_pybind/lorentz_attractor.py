from odesolver.flow_equations import FlowEquations


class LorentzAttractor(FlowEquations):
    def __init__(self):
        from lorentzattractorsimulation import LorentzAttractorFlow, LorentzAttractorJacobians
        super().__init__(flow=LorentzAttractorFlow(), jacobians=LorentzAttractorJacobians())

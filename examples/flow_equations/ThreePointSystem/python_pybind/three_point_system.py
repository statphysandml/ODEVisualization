from odesolver.flow_equations import FlowEquations


class ThreePointSystem(FlowEquations):
    def __init__(self):
        from threepointsystemsimulation import ThreePointSystemFlow, ThreePointSystemJacobians
        super().__init__(flow=ThreePointSystemFlow(), jacobians=ThreePointSystemJacobians())

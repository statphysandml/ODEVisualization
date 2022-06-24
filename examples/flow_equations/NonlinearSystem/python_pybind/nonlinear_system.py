from odesolver.flow_equations import FlowEquations


class NonlinearSystem(FlowEquations):
    def __init__(self):
        from nonlinearsystemsimulation import NonlinearSystemFlow, NonlinearSystemJacobians
        super().__init__(flow=NonlinearSystemFlow(), jacobians=NonlinearSystemJacobians())

from odesolver.vfcoor import VFCoor


class FlowEquations:
    def __init__(self, flow, jacobians):
        self._flow = flow
        self._jacobians = jacobians

    @property
    def dim(self):
        return self._flow.dim()

    @property
    def model(self):
        return self._flow.model

    @property
    def flow_variable(self):
        return self._flow.flow_variable

    @property
    def flow_parameters(self):
        return self._flow.flow_parameters

    def flow(self, coordinates: VFCoor):
        from odevisualizationlib.flow import compute_flow
        return VFCoor(compute_flow(coordinates.raw(), self._flow))

    def jacobians(self, coordinates: VFCoor):
        from odevisualizationlib.flow import compute_jacobian_elements
        return VFCoor(compute_jacobian_elements(coordinates.raw(), self._jacobians))
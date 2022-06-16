class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []

    def step(self, action):
        self.states.append(self.current_state())
        self.actions.append(action)
        self.update_state(action)

    def update_state(self, action):
        raise NotImplementedError

    def current_state(self):
        raise NotImplementedError

    def rewards(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
    
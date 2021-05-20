class ControlScheduler:
    def __init__(self, init_max_dec_diff, dec_half_life):
        self.init_max_dec_diff = init_max_dec_diff
        self.dec_half_life = dec_half_life

    def max_dec_diff(self, idx):
        if isinstance(self.init_max_dec_diff, float) and isinstance(self.dec_half_life, float):
            return self.init_max_dec_diff * (0.5 ** (idx / self.dec_half_life))
        else:
            return self.init_max_dec_diff

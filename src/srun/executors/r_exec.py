class RExecutor:
    def __init__(self):
        self._r = None
        self._available = None

    @property
    def available(self):
        if self._available is None:
            try:
                import rpy2.robjects as ro

                self._r = ro.r
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def execute(self, code):
        if not self.available:
            return False, "R not available (install rpy2)", ""
        try:
            result = self._r(code)
            return True, str(result), ""
        except Exception as e:
            return False, f"R Error: {e}", ""

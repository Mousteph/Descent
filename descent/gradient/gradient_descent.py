from .helpers import partial, gradient


class GradientDescent:
    def __name__(self) -> str:
        return self.__class__.__name__
        
    def _check_max_iter(self, i: int, max_iter: int):
        if i >= max_iter:
            print(f"{self.__name__()}: Max iteration reached: {max_iter}, the method did not converged")

    def _check_norm(self, norm: float, detect_div: float):
        self.last_norm = norm

        if norm >= detect_div:
            print(f"{self.__name__()}: The methode diverged. Take a smaller mu.")

    def _set_report(self, cost: float, it: int):
        self.last_cost = cost
        self.last_nb_it = it
            
    def __init__(self):
        self.last_norm = None
        self.last_cost = None
        self.last_nb_it = None
        
        self.partial = partial
        self.gradient = gradient
        
    def get_report(self):
        print(f"{self.__name__()}: Number of iterations: {self.last_nb_it} | Last Cost: {self.last_cost}")
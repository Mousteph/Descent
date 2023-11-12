from time import time
from typing import Callable

class GradientDescent:
    def __name__(self) -> str:
        """
        Returns the name of the class.

        Returns:
            str: The name of the class.
        """

        return self.__class__.__name__
        
    def _check_max_iter(self, i: int, max_iter: int) -> None:
        """
        Checks if the maximum number of iterations has been reached.

        Args:
            i (int): The current iteration.
            max_iter (int): The maximum number of iterations.
        """

        if i >= max_iter:
            print(f"{self.__name__()}: Max iteration reached: {max_iter}, the method did not converged")

    def _check_norm(self, norm: float, detect_div: float) -> None:
        """
        Checks if the method has diverged.

        Args:
            norm (float): The current norm.
            detect_div (float): The divergence detection value.
        """

        self.last_norm = norm

        if norm >= detect_div:
            print(f"{self.__name__()}: The methode diverged. Take a smaller mu.")

    def _set_report(self, cost: float, it: int) -> None:
        """
        Sets the report with the given cost and iteration.

        Args:
            cost (float): The cost.
            it (int): The iteration.
        """

        self.last_cost = cost
        self.last_nb_it = it
    
    @staticmethod
    def calculate_time(f: Callable):
        """
        Decorator to calculate the execution time of a function.

        Args:
            f (Callable): The function whose execution time is to be calculated.

        Returns:
            Callable: The decorated function which will return the function result and also save the execution time.
        """

        def save_time(self, *args, **kwargs):
            """
            Inner function to save the execution time of the function 'f'.

            Args:
                self: The instance of the class where the decorated function is defined.
                *args: Variable length argument list for the function 'f'.
                **kwargs: Arbitrary keyword arguments for the function 'f'.

            Returns:
                The result of the function 'f'.
            """

            time_start = time()
            res = f(self, *args, **kwargs)
            time_end = time()
            self.__time = time_end - time_start
            return res

        return save_time
            
    def __init__(self) -> None:
        """
        Initializes the GradientDescent.
        """

        self.last_norm = None
        self.last_cost = None
        self.last_nb_it = None
        self.__time = -1
        
    def get_report(self) -> None:
        """
        Prints the report.
        """

        print(f"{self.__name__()}: Number of iterations: {self.last_nb_it} | "\
              f"Last Cost: {self.last_cost} | Time: {round(self.__time, 4)}s")
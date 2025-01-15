__all__ = [
    "Problem",
    "DataSet",
    "Function",
    "MfDataSet",
    "MfFunc",
    "QuadrupleDataset",
    "QuadrupleCartesianProd",
    "TripleDataset",
    "TripleCartesianProd",

    "IDE",
    "PDE",
    "TimePDE",

    "FPDE",
    "TimeFPDE",

    "PDEOperator",
    "PDEOperatorCartesianProd",

]

from .base import Problem
from .dataset_function import Function
from .dataset_general import DataSet
from .dataset_mf import MfDataSet, MfFunc
from .dataset_quadruple import QuadrupleDataset, QuadrupleCartesianProd
from .dataset_triple import TripleDataset, TripleCartesianProd
from .fpde import FPDE, TimeFPDE
from .ide import IDE
from .pde import PDE, TimePDE
from .pde_operator import PDEOperator, PDEOperatorCartesianProd

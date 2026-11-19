from scipy.stats import special_ortho_group

from synthetic_experiments.sym_manifold.sym_ops import Var, Constant
from synthetic_experiments.sym_manifold.sym_ops import custom, concat, elementwise, reshape
from synthetic_experiments.sym_manifold.sym_manifold import CoordSystem


from synthetic_experiments.TSLassoData import TSLassoData

from sympy import sin, cos
import numpy as np

def ellipsoid(a = 1, b = 1, c = 1):
    return{
        "d": 2,
        "D": 3,
        "A": Constant(np.array([
                [np.pi/2, 0.0],
                [0.0, np.pi]
                ])),
        "b": Constant(np.array([np.pi/2, np.pi])),
        "y": [[
            f"{a} * sin(h_0) * cos(h_1)",
            f"{b} * sin(h_0) * sin(h_1)",
            f"{c} * cos(h_0)"
            ]]
        }

def torus(R = 2.0, r = 1.0):
    return {
        "d": 2,
        "D": 3,
        "A": Constant(np.array([
                [2 * np.pi, 0.0],
                [0.0, 2 * np.pi]
                ])),
        "b": Constant(np.array([np.pi, np.pi])),
        "y": [[
            f"({R} + {r} * cos(h_1)) * cos(h_0)",
            f"({R} + {r} * cos(h_1)) * sin(h_0)",
            f"{r} * sin(h_1)"
            ]]
        }

def cylinder(R = 1, h = 1, alpha = 0):
    return {
        "d": 2,
        "D": 3,
        "A": Constant(np.array([
                [np.pi, 0.0],
                [0.0, 1.0]
                ])),
        "b": Constant(np.array([np.pi, 0])),
        "y": [[
            f"{R} * cos(h_0)",
            f"{R} * sin(h_0)",
            f"{h} / 2 * h_1"],
            [
            f"(1 + {alpha} * h_2**2) * h_0", 
            f"(1 + {alpha} * h_2**2) * h_1", 
            "h_2"]]
        }

def sphere3D(R):
    return {
        "d": 3,
        "D": 4,
        "A": Constant(np.array([
            [np.pi/2, 0, 0],
            [0, np.pi/2, 0],
            [0, 0, np.pi]
        ])),
        "b": Constant(np.array([np.pi/2,np.pi/2,np.pi])),
        "y": [[
            f"{R} * cos(h_0)",
            f"{R} * sin(h_0) * cos(h_1)",
            f"{R} * sin(h_0) * sin(h_1) * cos(h_2)",
            f"{R} * sin(h_0) * sin(h_1) * sin(h_2)",
            ]]
        }
    
def swiss_roll(length = 21, angle = (3 * np.pi / 2, 9* np.pi /2)):
    return {
        "d": 2,
        "D": 3,

        "A": Constant(np.array([
            [(angle[1] - angle[0])/2, 0.0],
            [0.0, length / 2]
        ])),

        "b": Constant(np.array([
            (angle[1] + angle[0])/2,
            length / 2 
        ])),

        "y": [[
            "h_0 * cos(h_0)",
            "h_1",
            "h_0 * sin(h_0)"
        ]]
    }

def torus3D():
    return {
        "d": 3,
        "D": 6,
        "A": Constant(np.diag([np.pi, np.pi, np.pi])),
        "b": Constant(np.zeros(3)),
        "y": [[
            "cos(h_0)", "sin(h_0)",
            "cos(h_1)", "sin(h_1)",
            "cos(h_2)", "sin(h_2)"
            ]]
        }
    

def sample_sphere(n_points=1000, dim=4):
    X = np.random.randn(n_points, dim)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X   

def lift_to_higher_dim(data: np.ndarray, D: int, seed=None) -> np.ndarray:
    n, d = data.shape
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((D, d))
    Q, _ = np.linalg.qr(M)
    lifted_data = data @ Q.T
    return lifted_data

def directional_stretch(data, direction, scale=2.0, p=2.0, normalize=True):
    X = data.copy()
    v = direction / np.linalg.norm(direction)
    t = X @ v

    if normalize:
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = (t - t_min) / (t_max - t_min)
        else:
            t_norm = t.copy()
    else:
        t_norm = t
    f_t = scale * (t_norm ** p)
    if normalize:
        f_t = f_t * (t_max - t_min) + t_min
    X_new = X + (f_t - t)[:, None] * v

    return X_new

def sample_manifold(dict):
    A = dict["A"]
    b = dict["b"]
    d = dict["d"]
    D = dict["D"]
    y = dict["y"]
    
    R = Constant(special_ortho_group.rvs(48)[:, :D])

    x = Var(name="x", shape=(d, ))
    h = A @ x + b
    for i in range(len(y)):
        h = custom(h, y[i])
    y_small = h
    
    # y_small = custom(h, y)

    y_small_diff = y_small.to_custom().diff()
    
    small_coord_system = CoordSystem(d=d, D=D, 
                                     coord_domain=(-1, 1), coord_func=None, inv_func=y_small,
                                     coord_jac=None, inv_jac=y_small_diff)

    y = R @ y_small
    
    y_diff = (R @ y_small.to_custom()).diff()
    
    full_coord_system = CoordSystem(d=d, D=48, 
                                    coord_domain=(-1, 1), coord_func=None, inv_func=y,
                                    coord_jac=None, inv_jac=y_diff)

    return small_coord_system, full_coord_system

def sample_m1():

    # values used for paper experiment
    A = Constant(np.array([[-2.224596482726371, -0.07313352038651444],
                           [-0.13431766850153554, -1.0062494435296525]]))
    b = Constant(np.array([0.1244304928696931, 1.402807402444574]))

    # A = Constant(np.stack([np.array([np.random.uniform(-2.0, 2.0), np.random.uniform(-0.5, 0.5)]),
    #                        np.array([np.random.uniform(-np.pi/36, np.pi/36),
    #                                  (np.random.uniform(-np.pi/3, -np.pi/5) if np.random.rand() > 0.5 else
    #                                   np.random.uniform(np.pi/5, np.pi/3))])]))
    # b = Constant(np.array([np.random.uniform(-1.0, 1.0), np.random.uniform((5/12) * np.pi, (7/12) * np.pi)]))

    R = Constant(special_ortho_group.rvs(48)[:, :3])
    Rt = Constant(R.numpy_value.T)

    x = Var(name="x", shape=(2, ))
    o = Var(name="o", shape=(48, ))

    h = A @ x + b

    y_small = custom(h, ["1.0/(exp(h_0) + 1.0)", "cos(h_1)", "sin(2.0 * h_1)"])

    y_small_diff = y_small.to_custom().diff()

    small_coord_system = CoordSystem(d=2, D=3, coord_domain=(-1, 1), coord_func=None, inv_func=y_small,
                                     coord_jac=None, inv_jac=y_small_diff)

    y = R @ y_small
    y_diff = (R @ y_small.to_custom()).diff()

    full_coord_system = CoordSystem(d=2, D=48, coord_domain=(-1, 1), coord_func=None, inv_func=y,
                                    coord_jac=None, inv_jac=y_diff)

    return small_coord_system, full_coord_system


def sample_m2():

    # values used for paper experiment
    A = Constant(np.array([[-1.8851977431865239, 0.11138430303363234],
                           [0.006830803092539861, 1.2667849053051967]]))
    b = Constant(np.array([0.3702812303981724, -0.45769351026735927]))

    # A = Constant(np.random.random((2, 2)))
    # b = Constant(np.random.random((2,)))

    R = Constant(special_ortho_group.rvs(48)[:, :4])
    Rt = Constant(R.numpy_value.T)

    x = Var(name="x", shape=(2, ))
    o = Var(name="o", shape=(48, ))

    h1 = A @ x + b

    h2 = custom(h1, ["1.0/(1.0 + exp(h_0))", "exp(h_1)"])
    h3 = custom(h2, ["h_0 + 0.5", "2 * h_1"])
    h4 = custom(h3, ["h_0 * cos(h_1)", "h_0 * sin(h_1)"])

    y_small = concat([h2, h4])

    y_small_diff = y_small.to_custom().diff()

    small_coord_system = CoordSystem(d=2, D=4, coord_domain=(-1, 1), coord_func=None, inv_func=y_small,
                                     coord_jac=None, inv_jac=y_small_diff)

    y = R @ y_small
    y_diff = (R @ y_small.to_custom()).diff()

    full_coord_system = CoordSystem(d=2, D=48, coord_domain=(-1, 1), coord_func=None, inv_func=y,
                                    coord_jac=None, inv_jac=y_diff)

    return small_coord_system, full_coord_system


def sample_m3():

    # values used for paper experiment
    A = Constant(np.array([[-1.82073578, 0.08264555, 0.23420455],
                           [0.15310137, 1.0151491, -0.14543064],
                           [0.31639373, 0.30489013, 1.39837113]]))
    b = Constant(np.array([0.43044838, -0.61110428, -0.36894417]))

    # A = Constant(np.random.random((3, 3)))
    # b = Constant(np.random.random((3,)))

    R = Constant(special_ortho_group.rvs(48)[:, :7])
    Rt = Constant(R.numpy_value.T)

    x = Var(name="x", shape=(3,))
    o = Var(name="o", shape=(48,))

    h1 = A @ x + b

    h2 = custom(h1, ["1.0/(1.0 + exp(h_0))", "exp(h_1)", "log(1.0 + exp(h_2))"])
    h3 = custom(h2, ["h_1", "log(1.0 + h_0)", "h_0 + h_2", "h_1 + h_2"])
    h4 = custom(h3, ["h_1 * sin(h_3)", "h_0 * cos(h_2)", "cos(2 * h_2)", "sin(2 * h_3)"])

    y_small = concat([h2, h4])

    y_small_diff = y_small.to_custom().diff()

    small_coord_system = CoordSystem(d=3, D=7, coord_domain=(-1, 1), coord_func=None, inv_func=y_small,
                                     coord_jac=None, inv_jac=y_small_diff)

    y = R @ y_small

    y_diff = (R @ y_small.to_custom()).diff()

    full_coord_system = CoordSystem(d=3, D=48, coord_domain=(-1, 1), coord_func=None, inv_func=y,
                                    coord_jac=None, inv_jac=y_diff)

    return small_coord_system, full_coord_system

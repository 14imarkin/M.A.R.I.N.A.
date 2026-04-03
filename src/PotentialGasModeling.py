# Ivan P. Markin, 2026
# Моделирование газа со сложным потенциалом - двухчастичным вкладом Леннарда-Джонса и трехчастичным вкладом Аксельрода-Теллера.

# ------- data structures -------
# particle = [[x, y, z], [Vx, Vy, Vz]]
# particles = [particle 1, particle 2, ... , particle N]

# ------- imports -------
import numpy as np
from numba import njit, prange

# ------- hyperparameters -------
R = 0.05
m = 1.
SIGMA = 0.1
EPS = 0.1
C_at = 7e-10
DIFF_STEP = 1e-8

# ==========================================================
#                  ------- PHYSICS -------
# ==========================================================
# --- Lennard — Jones ---
@njit
def U_lj(p1, p2):
    """
    Классический двухчастичный потенциал Леннарда-Джонса.
    """
    r_norm = np.linalg.norm(p1[0] - p2[0])
    k = SIGMA / r_norm
    result = 4 * EPS * (k**12 - k**6)
    return result

@njit
def F_lj(p1, p2):
    """
    Расчет силы Леннарда-Джонса как градиент потенциальной 
    энергии через конечные центральные разности.
    """
    # dU/dx
    dx = np.array([[DIFF_STEP, 0., 0.], [0., 0., 0.]])
    p1_plus = p1 + dx
    p1_minus = p1 - dx
    Fx = ( U_lj(p1_plus, p2) - U_lj(p1_minus, p2) ) / ( 2 * DIFF_STEP )
    # dU/dy
    dy = np.array([[0., DIFF_STEP, 0.], [0., 0., 0.]])
    p1_plus = p1 + dy
    p1_minus = p1 - dy
    Fy = ( U_lj(p1_plus, p2) - U_lj(p1_minus, p2) ) / ( 2 * DIFF_STEP )
    # dU/dz
    dz = np.array([[0., 0., DIFF_STEP], [0., 0., 0.]])
    p1_plus = p1 + dz
    p1_minus = p1 - dz
    Fz = ( U_lj(p1_plus, p2) - U_lj(p1_minus, p2) ) / ( 2 * DIFF_STEP )
    # –grad(U_lj)
    result = (-1) * np.array([Fx, Fy, Fz])
    return result

# --- Axelrode — Teller ---
@njit
def U_at(p1, p2, p3):
    """
    Вычисление трехчастичного потенциала Аксельрод-Теллер
    """
    # Треугольник из 3-ех частиц
    r_12 = p2[0] - p1[0]
    r_12_norm = np.linalg.norm(r_12)
    r_23 = p3[0] - p2[0]
    r_23_norm = np.linalg.norm(r_23)
    r_31 = p1[0] - p3[0]
    r_31_norm = np.linalg.norm(r_31)
    # Косинусы углов треугольника
    cos_a1 = ( r_12 @ ( (-1) * r_31 ) ) / (r_12_norm * r_31_norm)
    cos_a2 = ( r_23 @ ( (-1) * r_12 ) ) / (r_12_norm * r_23_norm)
    cos_a3 = ( r_31 @ ( (-1) * r_23 ) ) / (r_23_norm * r_31_norm)
    # Конечный потенциал
    result = C_at * ( 1 + 3 * cos_a1 * cos_a2 * cos_a3) / ( (r_12_norm * r_23_norm * r_31_norm) ** 3 )
    return result

@njit
def F_at(p1, p2, p3):
    """
    Расчет силы Акссельрода-Теллера как
    градиент потенциальной энергии через
    конечные центральные разности.
    """
    # dU/dx
    dx = np.array([[DIFF_STEP, 0., 0.], [0., 0., 0.]])
    p1_plus = p1 + dx
    p1_minus = p1 - dx
    Fx = ( U_at(p1_plus, p2, p3) - U_at(p1_minus, p2, p3) )              / ( 2 * DIFF_STEP )
    # dU/dy
    dy = np.array([[0., DIFF_STEP, 0.], [0., 0., 0.]])
    p1_plus = p1 + dy
    p1_minus = p1 - dy
    Fy = ( U_at(p1_plus, p2, p3) - U_at(p1_minus, p2, p3) )              / ( 2 * DIFF_STEP )
    # dU/dz
    dz = np.array([[0., 0., DIFF_STEP], [0., 0., 0.]])
    p1_plus = p1 + dz
    p1_minus = p1 - dz
    Fz = ( U_at(p1_plus, p2, p3) - U_at(p1_minus, p2, p3) )              / ( 2 * DIFF_STEP )
    # –grad(U_at)
    result = (-1) * np.array([Fx, Fy, Fz])
    return result

# ==========================================================
#                ------- MODELING -------
# ==========================================================
# ------- accelerations -------
@njit
def compute_accelerations(particles):
    """
    Вычисление ускорений по потенциалам
    """
    N = len(particles)
    acc = np.zeros((N, 3))
    # Парные силы Леннарда-Джонса
    for i in prange(N):
        for j in prange(i+1, N):
            f_ij = F_lj(particles[i], particles[j])
            acc[i] += f_ij
            acc[j] -= f_ij
    # Трёхчастичные силы Аксельрода–Теллера
    for i in prange(N):
        for j in prange(i+1, N):
            for k in prange(j+1, N):
                f_i = F_at(particles[i], particles[j], particles[k])
                f_j = F_at(particles[j], particles[k], particles[i])
                f_k = F_at(particles[k], particles[i], particles[j])
                acc[i] += f_i
                acc[j] += f_j
                acc[k] += f_k
    return acc

# ------- velocity Verlet -------
@njit(parallel=True)
def verlet_step(particles, dt):
    # Ускорения в начале шага
    a0 = compute_accelerations(particles)
    # Полушаг по скорости
    for i in prange(len(particles)):
        particles[i][1] += 0.5 * a0[i] * dt
    # Шаг по координатам с обновленными скоростями
    for i in prange(len(particles)):
        particles[i][0] += particles[i][1] * dt
    # Ускорения после перемещения
    a1 = compute_accelerations(particles)
    # Второй полушаг по скорости
    for i in prange(len(particles)):
        particles[i][1] += 0.5 * a1[i] * dt
    return particles

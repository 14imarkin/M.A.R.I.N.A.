import random

# Генерация 10 случайных частиц в шаре радиуса R
def GenerateParticle(R=0.5, V=1., N=10, d_min=0.1):
  '''
  inputs:
    R     - радиус шара, в котором генерируются частицы
    V     - максимальный модуль скорости частиц
    N     - число частиц
    d_min - минимальное расстояние между частицы
  outputs:
    particles - массив частиц, соответствующей заданной в проекте структуре данных particles.
  '''
    particles = np.zeros((N, 2, 3))
    for i in range(N):
        # Генерируем случайное направление
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        u = random.uniform(0, 1)
        # Генерируем случайный радиус 
        r = R * (u ** (1/3))
        # Преобразуем сферические координаты в Декартовы
        theta = np.arccos(costheta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        # Генерация случайных скоростей
        Vx = random.uniform(-V, V)
        Vy = random.uniform(-V, V)
        Vz = random.uniform(-V, V)
        # Проверка наложения с уже существующими частицами
        collision = False
        for p in particles:
            # Вычисляем расстояние между частицами
            dx = x - p[0][0]
            dy = y - p[0][1]
            dz = z - p[0][2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            if distance < d_min:
                collision = True
                break
        # Если наложения нет, добавляем частицу
        if not collision:
            particles[i] = [[x, y, z], [Vx, Vy, Vz]]
            placed = True
    return particles

# Набор датасета
def CreateDataset(N, T, dt):
    '''
  inputs:
    N - число примеров в датасете
    T - число шагов моделирования в данном примере
    dt - шаг моделирования
  outputs:
    Dataset - обучающая выборка в NumPy массиве
  '''
    Dataset = np.zeros((N, 2, 100, 2, 3))
    for i in range(N):
#       Код для отслеживания прогресса
#       if i % 1000 == 0:
#           print(i)
        x = GenerateParticle(N=100)
        y = x.copy()
        for _ in range(T):
            y = modeling(y, dt)

        object = np.array([x, y])
        Dataset[i] = object
    return Dataset

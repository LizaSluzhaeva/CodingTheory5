from functools import reduce
import numpy as np


def C(n, k):
    """
            Функция вычисяет число сочетаний из n по k
    """
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0


def Kronecker_multiplication(A, B):
    """
         произведение Кронекера: A * B = [a{i,j}*B]
    """
    # пустой двумерный массив с количеством столбцов равным произведению числа столбцов в A и в B
    result = np.empty((0, A.shape[1] * B.shape[1]), dtype=int)
    for row in A:
        # строим пустой двумерный массив с количеством строк как у B
        row_result = np.empty((B.shape[0], 0), dtype=int)
        for elem in row:
            row_result = np.concatenate([row_result, elem * B], axis=1)
        result = np.concatenate([result, row_result])
    return result


# H(m,i) = I(2^(m-i)) * H * I(2^(i-1)
def H(m, i):
    return Kronecker_multiplication(
        Kronecker_multiplication(
            np.eye(2 ** (m - i), dtype=int),
            np.array([[1, 1], [1, -1]])
        ),
        np.eye(2 ** (i - 1), dtype=int)
    )


def convert_integer_to_binary_list(i):
    """
    Преобразование целого числа в двоичный список
    """
    if i == 0:
        return [0]
    if i == 1:
        return [1]
    result = []
    while i != 1:
        result += [i % 2]
        i //= 2
    return result + [1]


class RMCode:

    def __init__(self, r, m):
        self.r = r
        self.m = m
        # число информационных (исходных) разрядов
        # сумма i от 0 до r чисел сочетаний из m по i
        self.k = reduce(lambda x, y: x + y, map(lambda i: C(m, i), range(r + 1))) if r != 0 else 0
        # длина закодированного сообщения
        self.n = 2 ** m
        # порождающую матрицу оставляем пустой, генерацию выносим в отдельный метод
        self.G = None
        # список матриц H для декодирования, генерацию выносим в отдельный метод
        self.Hs = []
        # количество битов, добавленных к битам, считанным из файла, чтобы суммарное количество делилось на k нацело
        self.__added_bits = None

    # генерация порождающей матрицы, делегирует выполнение приватной функции
    def generate_G(self):
        self.G = self.__generate_G(self.r, self.m)

    # генерация порождающий матрицы кода Рида-Маллера с параметрами r и m
    def __generate_G(self, r, m):
        # G(0,m) = [11..1]
        if r == 0:
            return np.ones((1, 2 ** m), dtype=int)
        # G(m,m) = [ G(m-1,m)
        #             0..01   ]
        if r == m:
            # строим вектор из нужного количества нулей и на конце единица
            e = np.zeros((1, 2 ** m), dtype=int)
            e[0][-1] = 1
            return np.concatenate([self.__generate_G(m - 1, m), e])
        # G(r,m) = [ G(r,m-1)  G(r,m-1)
        #             00..0   G(r-1,m-1) ]
        G1 = self.__generate_G(r, m - 1)
        G2 = self.__generate_G(r - 1, m - 1)
        upper = np.concatenate([G1, G1], axis=1)
        lower = np.concatenate([np.zeros(G2.shape, dtype=int), G2], axis=1)
        return np.concatenate([upper, lower])

    # кодирование сообщения посредством умножения на матрицу G
    def encode(self, a):
        return (a @ self.G) % 2

    # построение списка матриц H для декодирования
    def generate_Hs(self):
        for i in range(1, self.m + 1):
            self.Hs += [H(self.m, i)]

    # декодирование сообщения
    def decode(self, w):
        # заменим все 0 в w на -1
        ww = w.copy()
        for i in range(len(ww)):
            if ww[i] == 0:
                ww[i] = -1
        # умножаем w последовательно на все H из списка
        for h in self.Hs:
            ww = ww @ h
        # поиск индекса максимального по модулю элемента в w
        j = 0
        max_elem = ww[0]
        for i, elem in enumerate(ww):
            if abs(elem) > abs(max_elem):
                max_elem = elem
                j = i
        # преобразуем j в двоичное представление
        vj = convert_integer_to_binary_list(j)
        # дополняем vj нулями до длины k-1
        vj += [0 for _ in range(self.k - 1 - len(vj))]
        # добавлем в начало единицу в случае, если w[j]>0, и ноль в противном случае
        vj = [1] + vj if max_elem > 0 else [0] + vj
        return np.array(vj)


def main():
    code = RMCode(2, 4)
    code.generate_G()
    code.generate_Hs()
    a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print('a =', a)
    w = code.encode(a)
    print('w =', w)
    print('decode =', code.decode(w))


if __name__ == '__main__':
    main()

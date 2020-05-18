"""
Test what if a class without init

"""


class A:

    para_a = 10
    @staticmethod
    def f_a():
        print('a')

    @staticmethod
    def f_b():
        print(A.para_a)


def test():

    A.f_a()
    A.f_b()

    A.para_a = 20

    A.f_b()

    print('d')


if __name__ == '__main__':
    test()




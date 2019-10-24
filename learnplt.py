import numpy as np
import matplotlib.pyplot as plt

def a():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)

    plt.plot(X, C)
    plt.plot(X, S)
    plt.show()

def b():
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)
    X = np.linspace(-np.pi, np.pi, 1024, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
    plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")
    plt.xlim(-4., 4.)
    plt.ylim(-1., 1.)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks(np.linspace(-1.2, 1.2, 7, endpoint=True))
    plt.show()

def c():
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)
    X = np.linspace(-np.pi, np.pi, 1024, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-", label='cosine')
    plt.plot(X, S, color="green", linewidth=1.0, linestyle="-", label='sine')
    plt.legend(loc='upper right')
    plt.xlim(-4., 4.)
    plt.ylim(-1., 1.)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks(np.linspace(-1.2, 1.2, 7, endpoint=True))
    plt.show()

def d():
    n = 256
    X = np.linspace(-np.pi, np.pi, endpoint=True)
    Y = np.sin(2*X)
    plt.plot(X, Y+1, color='blue', alpha=1.00)
    plt.plot(X, Y-1, color='red', alpha=1.00)
    plt.show()

def e():
    n = 4028
    X = np.random.uniform(0, 1, n)
    Y = np.random.uniform(0, 1, n)
    T = np.arctan2(Y, X)
    plt.scatter(X, Y, s=15, c=X*Y, alpha=0.5)
    plt.show()

def f():
    n = 12
    x = np.arange(n)
    y = (1 - x/n)*np.random.uniform(0.5, 1.0, n)
    plt.bar(x, y, facecolor='#9999ff', edgecolor='white')
    plt.bar(x, -y, facecolor='#ff9999', edgecolor='white')
    for a, b in zip(x, y):
        plt.text(a, b + 0.02, '%.2f'%b, ha='center', va='bottom')
        plt.text(a, -b-0.02, '%.2f'%-b, ha='center', va='bottom')
    
    plt.show()

def g():
    n = 1024
    def f(x, y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, f(X, Y), alpha=0.5, camp='hot')
    # C = plt.contour(X, Y, f(X, Y), 64, colors='black', linewidth=0.5)

    plt.show()

def h():
    n = 20
    z = np.random.uniform(0, 1, n)
    plt.pie(z)
    plt.show()

def i():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.05)
    Y = np.arange(-4, 4, 0.05)
    xx, yy = np.meshgrid(X, Y)
    R = np.sqrt(xx**2 + yy**2)
    z = np.sin(R)
    ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='hot')
    plt.show()

if __name__ == '__main__':
    i()
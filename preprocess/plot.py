import pylab as plt

# 房屋外边框画图
def boundary_plot(boundary):
    x_coords, y_coords = zip(*boundary[:, :2])
    x_coords += (x_coords[0],)
    y_coords += (y_coords[0],)
    plt.plot(x_coords, y_coords, marker='o')
    plt.axis('equal')
    plt.show()
import random
import numpy as np
import matplotlib.pyplot as plt

def planning_domain():
    x = 100 * np.random.rand()
    y = 100 * np.random.rand()
    return x, y

def steer(from_node, to_node, step_size=1.0):
    fx, fy = from_node
    tx, ty = to_node
    dx, dy = tx - fx, ty - fy
    dist = np.hypot(dx, dy)
    
    new_x = fx + (dx / dist) * step_size
    new_y = fy + (dy / dist) * step_size
    return new_x, new_y

def rrt(K, step_size=1.0, start=(50,50)):
    nodes = [start]
    parents = [-1]
    for _ in range(K):
        q_rand = planning_domain()
        idx_near = nearest_index(nodes, q_rand)
        q_new = steer(nodes[idx_near], q_rand, step_size)
        nodes.append(q_new)
        parents.append(idx_near)
    return nodes, parents


def nearest_index(nodes, point):
    xs = np.array([n[0] for n in nodes])
    ys = np.array([n[1] for n in nodes])
    px, py = point
    d2 = (xs - px)**2 + (ys - py)**2
    return int(np.argmin(d2))

    

if __name__ == "__main__":
    nodes, parents = rrt(K=2000, step_size=1.0)

    # draw tree
    for i in range(1, len(nodes)):
        p = parents[i]
        x1, y1 = nodes[p]
        x2, y2 = nodes[i]
        plt.plot([x1, x2], [y1, y2], '-', linewidth=0.8)

    xs = [x for x, _ in nodes]
    ys = [y for _, y in nodes]
    #circle1 = plt.Circle((50, 50), 20, color='r')

    #plt.gca().add_patch(circle1)
    plt.scatter(xs, ys, s=10)
    plt.xlim(0, 100); plt.ylim(0, 100)
    plt.title("RRT (step=1)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.show()
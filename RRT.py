import numpy as np
import matplotlib.pyplot as plt

def planning_domain(bounds=(0,100,0,100)):
    xmin, xmax, ymin, ymax = bounds
    return (np.random.uniform(xmin, xmax),
            np.random.uniform(ymin, ymax))

def steer(from_node, to_node, step_size=1.0):
    fx, fy = from_node
    tx, ty = to_node
    dx, dy = tx - fx, ty - fy
    dist = np.hypot(dx, dy)
    if dist == 0:
        return from_node
    return (fx + (dx/dist) * step_size, fy + (dy/dist) * step_size)

def nearest_index(nodes, point):
    xs = np.array([n[0] for n in nodes])
    ys = np.array([n[1] for n in nodes])
    px, py = point
    d2 = (xs - px)**2 + (ys - py)**2
    return int(np.argmin(d2))

def create_obstacles(num_obstacles, bounds=(0,100,0,100), r_range=(5,15), avoid_points=()):
    xmin, xmax, ymin, ymax = bounds
    obstacles = []
    for _ in range(num_obstacles):
        r = np.random.uniform(*r_range)
        while True:
            cx = np.random.uniform(xmin, xmax)
            cy = np.random.uniform(ymin, ymax)
            ok = True
            for p in avoid_points:
                if (cx - p[0])**2 + (cy - p[1])**2 <= r*r:
                    ok = False
                    break
            if ok:
                break
        obstacles.append(((cx, cy), r))
    return obstacles

def point_in_any_circle(p, obstacles):
    px, py = p
    for (cx, cy), r in obstacles:
        if (px - cx)**2 + (py - cy)**2 <= r*r:
            return True
    return False

def goal():
    pass

#make me a function that implements a goal that if the q_new sees the goal without being blocked by an obstacle, it connects to the goal
def connect_to_goal(q_new, goal, obstacles):
    if point_in_any_circle(goal, obstacles):
        return False
    # Check if the line from q_new to goal intersects any obstacle
    for (cx, cy), r in obstacles:
        # Line equation parameters
        x1, y1 = q_new
        x2, y2 = goal
        dx, dy = x2 - x1, y2 - y1
        a = dx*dx + dy*dy
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx)**2 + (y1 - cy)**2 - r*r
        discriminant = b*b - 4*a*c
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return False
    return True

def connect_to_goal2(q_new, goal, obstacles):
    if point_in_any_circle(goal, obstacles):
        return False
    for (cx, cy), r in obstacles:
        x1, y1 = q_new
        x2, y2 = goal
        dx, dy = x2 - x1, y2 - y1
        m = dy / dx
        b = y1 - m * x1
        a_prime = 1 + m**2
        b_prime = 2 * (m * (b - cx) - cy)
        c_prime = (cx)**2 + (b - cy)**2 - r*r
        discriminant = b_prime*b_prime - 4*a_prime*c_prime
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2*a_prime)
            t2 = (-b + np.sqrt(discriminant)) / (2*a_prime)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return False
    return True
    

def rrt(K, step_size=1.0, num_obstacles=6, start = (50,50), goal= (90,90), bounds=(0,100,0,100), r_range=(5,15)):
    nodes = [start]
    parents = [-1]
    obstacles = create_obstacles(num_obstacles, bounds, r_range, avoid_points=[start])
    if point_in_any_circle(start, obstacles):
        raise ValueError("start inside obstacle")
    if connect_to_goal(start, goal, obstacles):
        nodes.append(goal)
        parents.append(0)
        return nodes, parents, obstacles
    accepted = 0
    xmin, xmax, ymin, ymax = bounds
    while accepted < K:
        q_rand = planning_domain(bounds)
        if point_in_any_circle(q_rand, obstacles):
            continue
        i_near = nearest_index(nodes, q_rand)
        q_new = steer(nodes[i_near], q_rand, step_size)
        q_new = (np.clip(q_new[0], xmin, xmax),
                 np.clip(q_new[1], ymin, ymax))
        if connect_to_goal(q_new, goal, obstacles):
            nodes.append(goal)
            parents.append(len(nodes)-1)
            return nodes, parents, obstacles
        if point_in_any_circle(q_new, obstacles):
            continue
        nodes.append(q_new)
        parents.append(i_near)
        accepted += 1
    return nodes, parents, obstacles

if __name__ == "__main__":

    nodes, parents, obstacles = rrt(K=1000, step_size=1.0, num_obstacles=8, start=(15,10), goal=(90,90))
    for (cx, cy), r in obstacles:
        circle = plt.Circle((cx, cy), r, color='r', alpha=0.5)
        plt.gca().add_artist(circle)

    for i in range(1, len(nodes)):
        p = parents[i]
        x1, y1 = nodes[p]
        x2, y2 = nodes[i]
        plt.plot([x1, x2], [y1, y2], '-', linewidth=0.6)
        if nodes[i] == (90,90):
            plt.plot([x1, x2], [y1, y2], '-', linewidth=2)
    xs = [x for x, _ in nodes]
    ys = [y for _, y in nodes]
    plt.scatter(xs, ys, s=10)
    plt.scatter([nodes[0][0]], [nodes[0][1]], s=50, marker='o')
    plt.gca().set_aspect('equal', 'box')
    plt.xlim(0,100); plt.ylim(0,100)
    plt.title("RRT with circular obstacles")
    plt.legend()
    plt.show()

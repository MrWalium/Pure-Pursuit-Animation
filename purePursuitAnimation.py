import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from matplotlib.widgets import Button
from functools import partial
from IPython import display

plt.rcParams['animation.writer'] = 'ffmpeg'

path1 = [[0.0, 0.0], [0.571194595265405, -0.4277145118491421], [1.1417537280142898, -0.8531042347260006],
         [1.7098876452457967, -1.2696346390611464], [2.2705328851607995, -1.6588899151216996],
         [2.8121159420106827, -1.9791445882187304], [3.314589274316711, -2.159795566252656],
         [3.7538316863009027, -2.1224619985315876], [4.112485112342358, -1.8323249172947023],
         [4.383456805594431, -1.3292669972090994], [4.557386228943757, -0.6928302521681386],
         [4.617455513800438, 0.00274597627737883], [4.55408382321606, 0.6984486966257434],
         [4.376054025556597, 1.3330664239172116], [4.096280073621794, 1.827159263675668],
         [3.719737492364894, 2.097949296701878], [3.25277928312066, 2.108933125822431],
         [2.7154386886417314, 1.9004760368018616], [2.1347012144725985, 1.552342808106984],
         [1.5324590525923942, 1.134035376721349], [0.9214084611203568, 0.6867933269918683],
         [0.30732366808208345, 0.22955002391894264], [-0.3075127599907512, -0.2301742560363831],
         [-0.9218413719658775, -0.6882173194028102], [-1.5334674079795052, -1.1373288016589413],
         [-2.1365993767877467, -1.5584414896876835], [-2.7180981380280307, -1.9086314914221845],
         [-3.2552809639439704, -2.1153141204181285], [-3.721102967810494, -2.0979137913841046],
         [-4.096907306768644, -1.8206318841755131], [-4.377088212533404, -1.324440752295139],
         [-4.555249804461285, -0.6910016662308593], [-4.617336323713965, 0.003734984720118972],
         [-4.555948690867849, 0.7001491248072772], [-4.382109193278264, 1.3376838311365633],
         [-4.111620918085742, 1.8386823176628544], [-3.7524648889185794, 2.1224985058331005],
         [-3.3123191098095615, 2.153588702898333], [-2.80975246649598, 1.9712114570096653],
         [-2.268856462266256, 1.652958931009528], [-1.709001159778989, 1.2664395490411673],
         [-1.1413833971013372, 0.8517589252820573], [-0.5710732645795573, 0.4272721367616211], [0, 0],
         [0.571194595265405, -0.4277145118491421]]


class Animation:
    def __init__(self):
        # set this to true if you use rotations
        self.using_rotation = False

        # this determines how long (how many frames) the animation will run. 400 frames takes around 30 seconds.
        self.num_of_frames = 400

        self.robot = Robot()

        # the code below is for animation
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # for the sake of my sanity
        self.pi = np.pi
        # animation
        self.fig = plt.figure()
        self.trail_line = plt.plot([], '-', color='orange', linewidth=4)[0]

        # other setup, stationary stuff for example
        # plt.plot([initX], [initY], 'x',color='red',markersize=10)
        # plt.plot([path1[-1][0]], [path1[-1][1]], 'x',color='red',markersize=10)
        path_for_graph = np.array(convert_poses(path1))
        self.path = plt.plot(path_for_graph[:, 0], path_for_graph[:, 1], '--', color='grey')
        # plt.plot(pathForGraph[:, 0], pathForGraph[:, 1], 'o', color='purple', markersize=2)

        self.highlight_waypoint = plt.plot(0, 0, "o", color="yellow")[0]
        self.highlight_waypoint.set_visible(False)
        self.heading_line = plt.plot([], '-', color='red')[0]
        self.connection_line = plt.plot([], '-', color='green')[0]
        self.pose = plt.plot([], 'o', color='black', markersize=10)[0]

        plt.axis("scaled")
        plt.xlim(-6, 6)
        plt.ylim(-4, 4)
        self.trail_line_x = [self.robot[0]]
        self.trail_line_y = [self.robot[1]]

    def showAnimation(self):
        bound_func = partial(pure_pursuit_animation, robot=self.robot, heading_line=self.heading_line, pose=self.pose,
                             connection_line=self.connection_line, trail_line_x=self.trail_line_x,
                             trail_line_y=self.trail_line_y, trail_line=self.trail_line,
                             pi=self.pi, using_rotation=self.using_rotation)
        anim = animation.FuncAnimation(self.fig, bound_func, frames=self.num_of_frames, interval=50)

        # callback = Buttons()
        #
        # pos_add_waypoints = plt.axes([0.61, 0.01, 0.175, 0.075])
        # add_waypoints = Button(pos_add_waypoints, 'Add Waypoints')
        # add_waypoints.on_clicked(callback.addWayPoints)
        #
        # pos_reset = plt.axes([0.8, 0.01, 0.1, 0.075])
        # button_reset = Button(pos_reset, 'Reset')
        # button_reset.on_clicked(callback.reset)
        #
        # pos_clear_trail = plt.axes([0.21, 0.01, 0.13, 0.075])
        # button_clear_trail = Button(pos_clear_trail, 'Clear Trail')
        # button_clear_trail.on_clicked(callback.clear_trajectory_lines)
        #
        # pos_remove_waypoints = plt.axes([0.36, 0.01, 0.23, 0.075])
        # remove_waypoints = Button(pos_remove_waypoints, 'Remove Waypoints')
        # remove_waypoints.on_clicked(callback.remove_waypoint)
        #
        # self.fig.canvas.mpl_connect("button_press_event", callback.on_mouse_click)
        # self.fig.canvas.mpl_connect('motion_notify_event', callback.when_dragging)
        # self.fig.canvas.mpl_connect('button_release_event', callback.on_release)

        # video = anim.to_html5_video()
        # html = display.HTML(video)
        # display.display(html)

        plt.show()
        plt.close()


class Buttons:
    def __init__(self):
        self.a = 1


class Pure_Pursuit:
    def pt_to_pt_distance(self, pt1, pt2):
        distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        return distance

    # returns -1 if num is negative, 1 otherwise
    def sgn(self, num):
        if num >= 0:
            return 1
        else:
            return -1

    # this function needs to return 3 things IN ORDER: goalPt, lastFoundIndex, turnVel
    # think about this function as a snapshot in a while loop
    # given all information about the robot's current state, what should be the goalPt, lastFoundIndex, and turnVel?
    # the LFindex takes in the value of lastFoundIndex as input. Looking at it now I can't remember why I have it.
    # it is this way because I don't want the global lastFoundIndex to get modified in this function, instead, this function returns the updated lastFoundIndex value
    # this function will be feed into another function for creating animation
    def pure_pursuit_step(self, path, robot, pi):
        # extract currentX and currentY
        current_x = robot[0]
        current_y = robot[1]

        # use for loop to search intersections
        last_found_index = robot.last_found_index
        intersect_found = False

        starting_index = last_found_index

        for i in range(starting_index, len(path) - 1):

            # beginning of line-circle intersection code
            x1 = path[i][0] - current_x
            y1 = path[i][1] - current_y
            x2 = path[i + 1][0] - current_x
            y2 = path[i + 1][1] - current_y
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx ** 2 + dy ** 2)
            D = x1 * y2 - x2 * y1
            discriminant = (robot.look_ahead_dist ** 2) * (dr ** 2) - D ** 2

            if discriminant >= 0:
                sol_x1 = (D * dy + self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_x2 = (D * dy - self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2

                sol_pt1 = [sol_x1 + current_x, sol_y1 + current_y]
                sol_pt2 = [sol_x2 + current_x, sol_y2 + current_y]
                # end of line-circle intersection code

                min_x = min(path[i][0], path[i + 1][0])
                min_y = min(path[i][1], path[i + 1][1])
                max_x = max(path[i][0], path[i + 1][0])
                max_y = max(path[i][1], path[i + 1][1])

                # if one or both of the solutions are in range
                if ((min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)) or (
                        (min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)):

                    # make the decision by compare the distance between the intersections and the next point in path
                    if not (min_x <= sol_pt1[1] <= max_x) and (min_y <= sol_pt1[1] <= max_y) or self.pt_to_pt_distance(
                            sol_pt1, path[i + 1]) < self.pt_to_pt_distance(sol_pt2, path[i + 1]):
                        goal_pt = sol_pt1
                    else:
                        goal_pt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if self.pt_to_pt_distance(goal_pt, path[i + 1]) < self.pt_to_pt_distance([current_x, current_y],
                                                                                             path[i + 1]):
                        # update lastFoundIndex and exit
                        last_found_index = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment,
                        # but we also don't want it to go backward
                        last_found_index = i + 1

                # if no solutions are in range
                else:
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goal_pt = [path[last_found_index][0], path[last_found_index][1]]
        else:
            try:
                goal_pt
            except NameError:
                goal_pt = path1[0]
        # obtained goal point, now compute turn vel
        # initialize proportional controller constant
        Kp = 3

        # calculate absTargetAngle with the atan2 function
        abs_target_angle = math.atan2(goal_pt[1] - robot[1], goal_pt[0] - robot[0]) * 180 / pi
        if abs_target_angle < 0: abs_target_angle += 360

        # compute turn error by finding the minimum angle
        turn_error = abs_target_angle - robot.current_heading
        if turn_error > 180 or turn_error < -180:
            turn_error = -1 * self.sgn(turn_error) * (360 - abs(turn_error))

        # apply proportional controller
        turn_vel = Kp * turn_error

        return goal_pt, last_found_index, turn_vel


class Robot:
    def __init__(self):
        self.current_pos = [0, 0]
        self.current_heading = 330
        self.last_found_index = 0
        self.next_point_ndx = 1
        self.look_ahead_dist = 0.8
        self.linear_vel = 100
        self.dt = 50

    def __getitem__(self, item):
        return self.current_pos[0] if item == 0 else self.current_pos[1]

    def __setitem__(self, key, value):
        self.current_pos[key] = value

class Waypoint:
    def __init__(self, x=0, y=0, is_anchor=False):
        self.x = x
        self.y = y
        self.is_anchor = is_anchor

    def __getitem__(self, item):
        return self.x if item == 0 else self.y

    def get_pos(self):
        return [self.x, self.y]


def convert_poses(waypoints):
    return [waypoint.get_pos() for waypoint in waypoints]


def pure_pursuit_animation(frame, robot: Robot, heading_line, pose, connection_line, trail_line_x, trail_line_y,
                           trail_line, pi, using_rotation):
    # for the animation to loop
    if robot.last_found_index >= len(path1) - 2: robot.last_found_index = 0
    # if len(path1) > 0:
    #     print(f"{lastFoundIndex}, {path1[0]}")

    # call pure_pursuit_step to get info
    if len(path1) > 0:
        goal_pt, last_found_index, turn_vel = Pure_Pursuit().pure_pursuit_step(
            path1, robot, pi
        )

        # model: 200rpm drive with 18" width
        #               rpm   /s  circ   feet
        max_lin_vel_feet = 200 / 60 * pi * 4 / 12
        #               rpm   /s  center angle   deg
        max_turn_vel_deg = 200 / 60 * pi * 4 / 9 * 180 / pi

        # update x and y, but x and y stays constant here
        step_dis = robot.linear_vel / 100 * max_lin_vel_feet * robot.dt / 1000
        robot[0] += step_dis * np.cos(robot.current_heading * pi / 180)
        robot[1] += step_dis * np.sin(robot.current_heading * pi / 180)

        heading_line.set_data([robot[0], robot[0] + 0.5 * np.cos(robot.current_heading / 180 * pi)],
                              [robot[1], robot[1] + 0.5 * np.sin(robot.current_heading / 180 * pi)])
        connection_line.set_data([robot[0], goal_pt[0]], [robot[1], goal_pt[1]])

        robot.current_heading += turn_vel / 100 * max_turn_vel_deg * robot.dt / 1000
        if not using_rotation:
            robot.current_heading = robot.current_heading % 360
            if robot.current_heading < 0: robot.current_heading += 360

        # rest of the animation code
        trail_line_x.append(robot[0])
        trail_line_y.append(robot[1])

        pose.set_data([robot[0]], [robot[1]])
        trail_line.set_data(trail_line_x, trail_line_y)


def main():
    global path1
    path1 = [Waypoint(x, y) for x, y in path1]
    animationObj = Animation()
    animationObj.showAnimation()


main()
plt.close()

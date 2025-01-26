import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from matplotlib.widgets import Button
from functools import partial
from IPython import display

plt.rcParams['animation.writer'] = 'ffmpeg'


class Animation:
    def __init__(self):
        self.path1 = [[0.0, 0.0], [0.571194595265405, -0.4277145118491421], [1.1417537280142898, -0.8531042347260006],
                      [1.7098876452457967, -1.2696346390611464], [2.2705328851607995, -1.6588899151216996],
                      [2.8121159420106827, -1.9791445882187304], [3.314589274316711, -2.159795566252656],
                      [3.7538316863009027, -2.1224619985315876], [4.112485112342358, -1.8323249172947023],
                      [4.383456805594431, -1.3292669972090994], [4.557386228943757, -0.6928302521681386],
                      [4.617455513800438, 0.00274597627737883, True], [4.55408382321606, 0.6984486966257434],
                      [4.376054025556597, 1.3330664239172116], [4.096280073621794, 1.827159263675668],
                      [3.719737492364894, 2.097949296701878], [3.25277928312066, 2.108933125822431],
                      [2.7154386886417314, 1.9004760368018616], [2.1347012144725985, 1.552342808106984],
                      [1.5324590525923942, 1.134035376721349], [0.9214084611203568, 0.6867933269918683],
                      [0.30732366808208345, 0.22955002391894264], [-0.3075127599907512, -0.2301742560363831],
                      [-0.9218413719658775, -0.6882173194028102], [-1.5334674079795052, -1.1373288016589413],
                      [-2.1365993767877467, -1.5584414896876835], [-2.7180981380280307, -1.9086314914221845],
                      [-3.2552809639439704, -2.1153141204181285], [-3.721102967810494, -2.0979137913841046],
                      [-4.096907306768644, -1.8206318841755131], [-4.377088212533404, -1.324440752295139],
                      [-4.555249804461285, -0.6910016662308593], [-4.617336323713965, 0.003734984720118972, True],
                      [-4.555948690867849, 0.7001491248072772], [-4.382109193278264, 1.3376838311365633],
                      [-4.111620918085742, 1.8386823176628544], [-3.7524648889185794, 2.1224985058331005],
                      [-3.3123191098095615, 2.153588702898333], [-2.80975246649598, 1.9712114570096653],
                      [-2.268856462266256, 1.652958931009528], [-1.709001159778989, 1.2664395490411673],
                      [-1.1413833971013372, 0.8517589252820573], [-0.5710732645795573, 0.4272721367616211], [0, 0]]
        # , [0, 0], [0.571194595265405, -0.4277145118491421]

        self.path1 = [Waypoint(point[0], point[1]) if len(point) == 2 else Waypoint(point[0], point[1], point[2]) for point in self.path1]

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
        path_for_graph = np.array(convert_poses(self.path1))
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
        bound_func = partial(pure_pursuit_animation, robot=self.robot, path1=self.path1, heading_line=self.heading_line,
                             pose=self.pose,
                             connection_line=self.connection_line, trail_line_x=self.trail_line_x,
                             trail_line_y=self.trail_line_y, trail_line=self.trail_line,
                             pi=self.pi, using_rotation=self.using_rotation)
        anim = animation.FuncAnimation(self.fig, bound_func, frames=self.num_of_frames, interval=50)

        callback = Buttons()

        pos_add_waypoints = plt.axes((0.61, 0.01, 0.175, 0.075))
        add_waypoints = Button(pos_add_waypoints, 'Add Waypoints')

        pos_reset = plt.axes((0.8, 0.01, 0.1, 0.075))
        button_reset = Button(pos_reset, 'Reset')

        pos_clear_trail = plt.axes((0.21, 0.01, 0.13, 0.075))
        button_clear_trail = Button(pos_clear_trail, 'Clear Trail')

        pos_remove_waypoints = plt.axes((0.36, 0.01, 0.23, 0.075))
        remove_waypoints = Button(pos_remove_waypoints, 'Remove Waypoints')

        add_waypoints.on_clicked(partial(callback.add_waypoints, b_remove_waypoints=remove_waypoints,
                                         bWaypoints=add_waypoints))
        button_reset.on_clicked(partial(callback.reset, path1=self.path1, path=self.path, xs=self.trail_line_x,
                                        ys=self.trail_line_y, trajectory_line=self.trail_line))
        button_clear_trail.on_clicked(partial(callback.clear_trajectory_lines, xs=self.trail_line_x,
                                              ys=self.trail_line_y, trajectory_line=self.trail_line))
        remove_waypoints.on_clicked(
            partial(callback.remove_waypoint, bWaypoints=add_waypoints, b_remove_waypoints=remove_waypoints))

        self.fig.canvas.mpl_connect("button_press_event",
                                    partial(callback.on_mouse_click, fig=self.fig, path1=self.path1,
                                            path=self.path, xs=self.trail_line_x,
                                            ys=self.trail_line_y,
                                            trajectory_line=self.trail_line))
        self.fig.canvas.mpl_connect('motion_notify_event', partial(callback.when_dragging, fig=self.fig,
                                                                   path1=self.path1,
                                                                   highlight_waypoint=self.highlight_waypoint,
                                                                   path=self.path,
                                                                   xs=self.trail_line_x, ys=self.trail_line_y,
                                                                   trajectory_line=self.trail_line))
        self.fig.canvas.mpl_connect('button_release_event', callback.on_release)

        # video = anim.to_html5_video()
        # html = display.HTML(video)
        # display.display(html)

        plt.show()
        plt.close()


class Buttons:
    def __init__(self):
        self.add_waypoints_pressed = False
        self.remove_waypoints_pressed = False
        self.is_dragging = False
        self.visible_waypoint_selector = -1
        self.waypoint_selector = -1
        self.INTERACTION_DIST = 0.5

    def add_waypoints(self, event, b_remove_waypoints, bWaypoints):
        self.add_waypoints_pressed = not self.add_waypoints_pressed

        if self.add_waypoints_pressed:
            if self.remove_waypoints_pressed:
                self.remove_waypoints_pressed = False
                self.change_button_colors(b_remove_waypoints)
            self.change_button_colors(bWaypoints, 'green', "lightgreen")
        else:
            self.change_button_colors(bWaypoints)

        plt.draw()

    def change_button_colors(self, button, color1="lightgrey", color2="white"):
        button.color = color1
        button.hovercolor = color2

    def remove_graph_waypoint(self, event, path1, path, xs, ys, trajectory_line):
        if len(path1) <= 1:
            self.reset_graph(path1, path, xs, ys, trajectory_line)
            return

        ndx, distance = self.find_nearest_waypoint([event.xdata, event.ydata], path1)
        if distance < self.INTERACTION_DIST:
            path1.pop(ndx)

    def add_graph_waypoint(self, event, path1):
        self.update_visible_selector(event, path1)
        if self.waypoint_selector == -1:
            path1.append(Waypoint(event.xdata, event.ydata))
        else:
            path1.insert(self.waypoint_selector + 1, Waypoint(event.xdata, event.ydata))
            self.waypoint_selector += 1

    def update_path(self, path1, path):
        # Convert path1 to a NumPy array of shape (N, 2)
        path_for_graph = np.array(convert_poses(path1))

        # Update the plot with the new path
        if len(path1) > 0:
            for line in path:
                line.set_data(path_for_graph[:, 0], path_for_graph[:, 1])  # Set the new path data

    def when_dragging(self, event, fig, path1, highlight_waypoint, path, xs, ys, trajectory_line):
        if event.inaxes == fig.axes[0]:
            if self.is_dragging:
                if self.add_waypoints_pressed:
                    self.add_graph_waypoint(event, path1)
                if self.remove_waypoints_pressed:
                    self.remove_graph_waypoint(event, path1, path, xs, ys, trajectory_line)

                self.update_path(path1, path)
            else:
                self.update_visible_selector(event, path1)
                if self.visible_waypoint_selector > -1:
                    highlight_waypoint.set_data([path1[self.visible_waypoint_selector][0]],
                                                [path1[self.visible_waypoint_selector][1]])
                    highlight_waypoint.set_visible(True)
                else:
                    highlight_waypoint.set_visible(False)

        plt.draw()

    def on_release(self, event):
        self.is_dragging = False

    def reset(self, event, path1, path, xs, ys, trajectory_line):
        self.reset_graph(path1, path, xs, ys, trajectory_line)

    def reset_graph(self, path1, path, xs, ys, trajectory_line):
        path1.clear()
        for line in path:
            line.set_data([], [])
        self.clear_graph_trajectory_lines(xs, ys, trajectory_line)
        plt.draw()

    def on_mouse_click(self, event, fig, path1, path, xs, ys, trajectory_line):
        if event.inaxes == fig.axes[0]:
            if event.button == 1:
                if self.add_waypoints_pressed:
                    # Append the clicked point as a tuple/list
                    # print(f"{event.xdata}, {event.ydata}")
                    self.is_dragging = True
                    if len(path1) > 0:
                        ndx, distance = self.find_nearest_waypoint([event.xdata, event.ydata], path1)
                        if distance < self.INTERACTION_DIST and ndx == 0:
                            path1.append(path1[0])
                        else:
                            self.add_graph_waypoint(event, path1)
                    else:
                        self.add_graph_waypoint(event, path1)  # Use append to add a new point as a list

                    self.update_path(path1, path)

                elif self.remove_waypoints_pressed:
                    # print(self.find_nearest_waypoint([event.xdata, event.ydata], path1))
                    self.is_dragging = True

                    self.remove_graph_waypoint(event, path1, path, xs, ys, trajectory_line)

                    self.update_path(path1, path)
            else:
                self.update_visible_selector(event, path1)
                if event.button == 2 or event.button == 3:
                    self.waypoint_selector = self.visible_waypoint_selector
            plt.draw()  # Ensure the plot updates after the click

    def update_visible_selector(self, event, path1):
        if len(path1) >= 1:
            ndx, distance = self.find_nearest_waypoint([event.xdata, event.ydata], path1)
            if distance < self.INTERACTION_DIST:
                self.visible_waypoint_selector = ndx
                # print(f"{self.visible_waypoint_selector}, {self.waypoint_selector}")
                return
        self.visible_waypoint_selector = -1
        # print(f"{self.visible_waypoint_selector}, {self.waypoint_selector}")

    def clear_trajectory_lines(self, event, xs, ys, trajectory_line):
        self.clear_graph_trajectory_lines(xs, ys, trajectory_line)
        plt.draw()

    def clear_graph_trajectory_lines(self, xs, ys, trajectory_line):
        xs.clear()
        ys.clear()
        trajectory_line.set_data([], [])

    def remove_waypoint(self, event, bWaypoints, b_remove_waypoints):
        self.remove_waypoints_pressed = not self.remove_waypoints_pressed

        if self.remove_waypoints_pressed:
            if self.add_waypoints_pressed:
                self.add_waypoints_pressed = False
                self.change_button_colors(bWaypoints)
            self.change_button_colors(b_remove_waypoints, "red", "lightcoral")
        else:
            self.change_button_colors(b_remove_waypoints)

        plt.draw()

    def find_nearest_waypoint(self, current_position, waypoints):
        # Convert waypoints to a NumPy array if it's not already
        waypoints = np.array(convert_poses(waypoints))

        # Handle the edge case where there are no waypoints
        if len(waypoints) == 0:
            return -1

        # Calculate the distances between the current position and each waypoint
        distances = np.linalg.norm(waypoints - current_position, axis=1)

        # Find the index of the nearest waypoint
        nearest_index = int(np.argmin(distances))  # Ensure compatibility with Python lists

        return nearest_index, distances[nearest_index]


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
    """
        Find solutions (circle line intersection forumula)
        
        If there's solutions,
        
        Find the one that gives you the most progress towards the next point
        
        else go to the last found goal point.
    """
    def pure_pursuit_step(self, path, robot, pi):
        # use for loop to search intersections

        starting_index = robot.last_found_index

        goal_pts = list()

        for i in range(0, len(path) - 1):
            starting_index_incremented = increment_val(starting_index, 1, len(path) - 1)

            # beginning of line-circle intersection code
            x1 = path[starting_index][0] - robot[0]
            y1 = path[starting_index][1] - robot[1]
            x2 = path[starting_index_incremented][0] - robot[0]
            y2 = path[starting_index_incremented][1] - robot[1]
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx ** 2 + dy ** 2)
            D = x1 * y2 - x2 * y1
            discriminant = (robot.look_ahead_dist ** 2) * (dr ** 2) - D ** 2

            if discriminant >= 0 and dr != 0:
                sol_x1 = (D * dy + self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_x2 = (D * dy - self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2

                sol_pt1 = [sol_x1 + robot[0], sol_y1 + robot[1]]
                sol_pt2 = [sol_x2 + robot[0], sol_y2 + robot[1]]
                # end of line-circle intersection code

                min_x = min(path[starting_index][0], path[starting_index_incremented][0])
                min_y = min(path[starting_index][1], path[starting_index_incremented][1])
                max_x = max(path[starting_index][0], path[starting_index_incremented][0])
                max_y = max(path[starting_index][1], path[starting_index_incremented][1])

                next_point = starting_index_incremented if not path[robot.next_point_ndx].is_anchor \
                    else robot.next_point_ndx

                sol_pt1_in_range = (min_x <= sol_pt1[0] <= max_x) and (min_y <= sol_pt1[1] <= max_y)
                sol_pt2_in_range = (min_x <= sol_pt2[0] <= max_x) and (min_y <= sol_pt2[1] <= max_y)

                # if one or both of the solutions are in range
                if sol_pt1_in_range or sol_pt2_in_range:

                    # make the decision by compare the distance between the intersections and the next point in path
                    if not sol_pt2_in_range or (sol_pt1_in_range and
                             self.pt_to_pt_distance(sol_pt1, path[next_point]) <
                             self.pt_to_pt_distance(sol_pt2, path[next_point])):
                        goal_pt = sol_pt1
                    else:
                        goal_pt = sol_pt2

                    goal_pts.append([goal_pt[0], goal_pt[1]])

                    if self.pt_to_pt_distance(path[next_point], robot.current_pos) < robot.look_ahead_dist:
                        robot.next_point_ndx = increment_val(robot.next_point_ndx, 1, len(path) - 1)
                        robot.last_found_index = increment_val(robot.last_found_index, 1, len(path) - 1)

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if self.pt_to_pt_distance(goal_pt, path[next_point]) < self.pt_to_pt_distance(
                            robot.current_pos,
                            path[next_point]):
                        # update lastFoundIndex and exit
                        robot.last_found_index = starting_index
                        robot.next_point_ndx = increment_val(robot.last_found_index, 1, len(path) - 1)
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment,
                        # but we also don't want it to go backward
                        robot.last_found_index = starting_index_incremented
                        robot.next_point_ndx = increment_val(robot.last_found_index, 1, len(path) - 1)

                # if no solutions are in range
                else:
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goal_pt = [path[robot.last_found_index][0], path[robot.last_found_index][1]]

            if path[starting_index].is_anchor and self.pt_to_pt_distance(path[starting_index], robot.current_pos) > robot.look_ahead_dist:
                break

            starting_index = increment_val(starting_index, 1, len(path) - 1)

        try:
            goal_pt
        except NameError:
            goal_pt = path[0]

        if len(goal_pts) > 0:
            goal_pt = goal_pts[-1]

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

        return goal_pt, robot.last_found_index, turn_vel


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
    def __init__(self, x=0.0, y=0.0, is_anchor=False):
        self.x = x
        self.y = y
        self.is_anchor = is_anchor

    def __getitem__(self, item):
        return self.x if item == 0 else self.y

    def get_pos(self):
        return [self.x, self.y]


def convert_poses(waypoints):
    return [waypoint.get_pos() for waypoint in waypoints]


def pure_pursuit_animation(frame, robot: Robot, path1, heading_line, pose, connection_line, trail_line_x, trail_line_y,
                           trail_line, pi, using_rotation):
    # for the animation to loop
    if robot.last_found_index >= len(path1): robot.last_found_index = 0
    if robot.next_point_ndx >= len(path1): robot.next_point_ndx = 0
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


def increment_val(val, increment, max_val):
    return val + increment if val + increment <= max_val else 0


def main():
    animation_obj = Animation()
    animation_obj.showAnimation()


main()
plt.close()

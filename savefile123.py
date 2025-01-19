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

global currentPos, currentHeading, lastFoundIndex, lookAheadDis, linearVel, pi, fig, trajectory_lines, trajectory_line
global heading_lines, heading_line, connection_lines, connection_line, poses, pose, dt, xs, ys, using_rotation, path, highlight_waypoint

class Animation:
    def __init__(self):
        # THIS IS DIFFERENT THAN BEFORE! initialize variables here
        # you can also change the Kp constant which is located at line 113
        global currentPos, currentHeading, lastFoundIndex, lookAheadDis, linearVel, pi, fig, trajectory_lines
        global trajectory_line, heading_lines, heading_line, connection_lines, connection_line, poses, pose, dt, xs, ys, using_rotation, path, highlight_waypoint, next_point_ndx

        currentPos = [0, 0]
        currentHeading = 330
        lastFoundIndex = 0
        next_point_ndx = 1
        lookAheadDis = 0.8
        linearVel = 100

        # set this to true if you use rotations
        using_rotation = False

        # this determines how long (how many frames) the animation will run. 400 frames takes around 30 seconds.
        self.numOfFrames = 400

        # the code below is for animation
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # for the sake of my sanity
        pi = np.pi
        # animation
        fig = plt.figure()
        trajectory_line = plt.plot([], '-', color='orange', linewidth=4)[0]

        # other setup, stationary stuff for example
        # plt.plot([initX], [initY], 'x',color='red',markersize=10)
        # plt.plot([path1[-1][0]], [path1[-1][1]], 'x',color='red',markersize=10)
        self.pathForGraph = np.array(convert_poses(path1))
        path = plt.plot(self.pathForGraph[:, 0], self.pathForGraph[:, 1], '--', color='grey')
        # plt.plot(pathForGraph[:, 0], pathForGraph[:, 1], 'o', color='purple', markersize=2)

        highlight_waypoint = plt.plot(0, 0, "o", color="yellow")[0]
        highlight_waypoint.set_visible(False)
        heading_line = plt.plot([], '-', color='red')[0]
        connection_line = plt.plot([], '-', color='green')[0]
        pose = plt.plot([], 'o', color='black', markersize=10)[0]

        plt.axis("scaled")
        plt.xlim(-6, 6)
        plt.ylim(-4, 4)
        dt = 50
        xs = [currentPos[0]]
        ys = [currentPos[1]]


    def showAnimation(self):
        global bWaypoints, locBWaypoints, b_remove_waypoints
        anim = animation.FuncAnimation(fig, pure_pursuit_animation, frames=self.numOfFrames, interval=50)
        callback = Buttons()

        locBWaypoints = plt.axes([0.61, 0.01, 0.175, 0.075])
        bWaypoints = Button(locBWaypoints, 'Add Waypoints')
        bWaypoints.on_clicked(callback.addWayPoints)

        locBReset = plt.axes([0.8, 0.01, 0.1, 0.075])
        b_reset = Button(locBReset, 'Reset')
        b_reset.on_clicked(callback.reset)

        loc_clear_trail = plt.axes([0.21, 0.01, 0.13, 0.075])
        b_clear_trail = Button(loc_clear_trail, 'Clear Trail')
        b_clear_trail.on_clicked(callback.clear_trajectory_lines)

        loc_remove_waypoints = plt.axes([0.36, 0.01, 0.23, 0.075])
        b_remove_waypoints = Button(loc_remove_waypoints, 'Remove Waypoints')
        b_remove_waypoints.on_clicked(callback.remove_waypoint)

        fig.canvas.mpl_connect("button_press_event", callback.on_mouse_click)
        fig.canvas.mpl_connect('motion_notify_event', callback.when_dragging)
        fig.canvas.mpl_connect('button_release_event', callback.on_release)

        plt.show()
        # video = anim.to_html5_video()
        # html = display.HTML(video)
        # display.display(html)
        plt.close()

class PurePursuit:
    # helper functions
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
    def pure_pursuit_step(self, path, currentPos, currentHeading, lookAheadDis, LFindex):
        global lastFoundIndex, linearVel, pi, fig, trajectory_lines
        global trajectory_line, heading_lines, heading_line, connection_lines, connection_line, poses, pose, dt, xs, ys, using_rotation

        # extract currentX and currentY
        currentX = currentPos[0]
        currentY = currentPos[1]

        # use for loop to search intersections
        lastFoundIndex = LFindex
        intersectFound = False

        startingIndex = lastFoundIndex

        for i in range(startingIndex, len(path) - 1):

            # beginning of line-circle intersection code
            x1 = path[i][0] - currentX
            y1 = path[i][1] - currentY
            x2 = path[i + 1][0] - currentX
            y2 = path[i + 1][1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx ** 2 + dy ** 2)
            D = x1 * y2 - x2 * y1
            discriminant = (lookAheadDis ** 2) * (dr ** 2) - D ** 2

            if discriminant >= 0:
                sol_x1 = (D * dy + self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_x2 = (D * dy - self.sgn(dy) * dx * np.sqrt(discriminant)) / dr ** 2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr ** 2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr ** 2

                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
                # end of line-circle intersection code

                minX = min(path[i][0], path[i + 1][0])
                minY = min(path[i][1], path[i + 1][1])
                maxX = max(path[i][0], path[i + 1][0])
                maxY = max(path[i][1], path[i + 1][1])

                # print(i)

                # if one or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

                    foundIntersection = True

                    # make the decision by compare the distance between the intersections and the next point in path
                    if not (minX <= sol_pt1[1] <= maxX) and (minY <= sol_pt1[1] <= maxY) or self.pt_to_pt_distance(sol_pt1, path[i + 1]) < self.pt_to_pt_distance(sol_pt2, path[i + 1]):
                        goalPt = sol_pt1
                    else:
                        goalPt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if self.pt_to_pt_distance(goalPt, path[i + 1]) < self.pt_to_pt_distance([currentX, currentY], path[i + 1]):
                        # update lastFoundIndex and exit
                        lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        lastFoundIndex = i + 1

                # if no solutions are in range
                else:
                    foundIntersection = False
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]
        else:
            try:
                goalPt
            except NameError:
                goalPt = path1[0]
        # obtained goal point, now compute turn vel
        # initialize proportional controller constant
        Kp = 3

        # calculate absTargetAngle with the atan2 function
        absTargetAngle = math.atan2(goalPt[1] - currentPos[1], goalPt[0] - currentPos[0]) * 180 / pi
        if absTargetAngle < 0: absTargetAngle += 360

        # compute turn error by finding the minimum angle
        turnError = absTargetAngle - currentHeading
        if turnError > 180 or turnError < -180:
            turnError = -1 * self.sgn(turnError) * (360 - abs(turnError))

        # apply proportional controller
        turnVel = Kp * turnError

        return goalPt, lastFoundIndex, turnVel

class Buttons:
    def __init__(self):
        self.add_waypoints_pressed = False
        self.remove_waypoints_pressed = False
        self.is_dragging = False
        self.visible_waypoint_selector = -1
        self.waypoint_selector = -1
        self.INTERACTION_DIST = 0.5

    def addWayPoints(self, event):
        global bWaypoints, b_remove_waypoints
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

    def remove_graph_waypoint(self, event):
        if len(path1) <= 1:
            self.reset_graph()
            return

        ndx, distance = self.find_nearest_waypoint([event.xdata, event.ydata], path1)
        if distance < self.INTERACTION_DIST:
            path1.pop(ndx)

    def add_graph_waypoint(self, event):
        self.update_visible_selector(event)
        if self.waypoint_selector == -1:
            path1.append(Waypoint(event.xdata, event.ydata))
        else:
            path1.insert(self.waypoint_selector + 1, Waypoint(event.xdata, event.ydata))
            self.waypoint_selector += 1

    def update_path(self):
        # Convert path1 to a NumPy array of shape (N, 2)
        path_for_graph = np.array(convert_poses(path1))

        # Update the plot with the new path
        if len(path1) > 0:
            for line in path:
                line.set_data(path_for_graph[:, 0], path_for_graph[:, 1])  # Set the new path data

    def when_dragging(self, event):
        if event.inaxes == fig.axes[0]:
            if self.is_dragging:
                if self.add_waypoints_pressed:
                    self.add_graph_waypoint(event)
                if self.remove_waypoints_pressed:
                    self.remove_graph_waypoint(event)

                self.update_path()
            else:
                self.update_visible_selector(event)
                if self.visible_waypoint_selector > -1:
                    highlight_waypoint.set_data([path1[self.visible_waypoint_selector][0]], [path1[self.visible_waypoint_selector][1]])
                    highlight_waypoint.set_visible(True)
                else:
                    highlight_waypoint.set_visible(False)

        plt.draw()

    def on_release(self, event):
        self.is_dragging = False

    def reset(self, event):
        self.reset_graph()

    def reset_graph(self):
        global path, path1, trajectory_line, xs, ys
        path1 = []
        for line in path:
            line.set_data([], [])
        self.clear_graph_trajectory_lines()
        plt.draw()

    def on_mouse_click(self, event):
        global path1, path
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
                            self.add_graph_waypoint(event)
                    else:
                        self.add_graph_waypoint(event)  # Use append to add a new point as a list

                    self.update_path()

                elif self.remove_waypoints_pressed:
                    # print(self.find_nearest_waypoint([event.xdata, event.ydata], path1))
                    self.is_dragging = True

                    self.remove_graph_waypoint(event)

                    self.update_path()
            else:
                self.update_visible_selector(event)
                if event.button == 2 or event.button == 3:
                    self.waypoint_selector = self.visible_waypoint_selector
            plt.draw()  # Ensure the plot updates after the click

    def update_visible_selector(self, event):
        if len(path1) >= 1:
            ndx, distance = self.find_nearest_waypoint([event.xdata, event.ydata], path1)
            if distance < self.INTERACTION_DIST:
                self.visible_waypoint_selector = ndx
                #print(f"{self.visible_waypoint_selector}, {self.waypoint_selector}")
                return
        self.visible_waypoint_selector = -1
        #print(f"{self.visible_waypoint_selector}, {self.waypoint_selector}")

    def clear_trajectory_lines(self, event):
        self.clear_graph_trajectory_lines()
        plt.draw()

    def clear_graph_trajectory_lines(self):
        global xs, ys, trajectory_line
        xs = []
        ys = []
        trajectory_line.set_data([], [])

    def remove_waypoint(self, event):
        global b_remove_waypoints, bWaypoints
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

def pure_pursuit_animation(frame):
    # define globals
    global currentPos, currentHeading, lastFoundIndex, lookAheadDis, linearVel, pi, fig, trajectory_lines
    global trajectory_line, heading_lines, heading_line, connection_lines, connection_line, poses, pose, dt, xs, ys, using_rotation, next_point_ndx

    # for the animation to loop
    if lastFoundIndex >= len(path1) - 2: lastFoundIndex = 0
    # if len(path1) > 0:
    #     print(f"{lastFoundIndex}, {path1[0]}")

    # call pure_pursuit_step to get info
    if len(path1) > 0:
        goalPt, lastFoundIndex, turnVel = PurePursuit().pure_pursuit_step(
            path1, currentPos, currentHeading, lookAheadDis, lastFoundIndex
        )

        # print()

        # model: 200rpm drive with 18" width
        #               rpm   /s  circ   feet
        maxLinVelfeet = 200 / 60 * pi * 4 / (12)
        #               rpm   /s  center angle   deg
        maxTurnVelDeg = 200 / 60 * pi * 4 / 9 * 180 / pi

        # update x and y, but x and y stays constant here
        stepDis = linearVel / 100 * maxLinVelfeet * dt / 1000
        currentPos[0] += stepDis * np.cos(currentHeading * pi / 180)
        currentPos[1] += stepDis * np.sin(currentHeading * pi / 180)

        heading_line.set_data([currentPos[0], currentPos[0] + 0.5 * np.cos(currentHeading / 180 * pi)],
                              [currentPos[1], currentPos[1] + 0.5 * np.sin(currentHeading / 180 * pi)])
        connection_line.set_data([currentPos[0], goalPt[0]], [currentPos[1], goalPt[1]])

        currentHeading += turnVel / 100 * maxTurnVelDeg * dt / 1000
        if not using_rotation:
            currentHeading = currentHeading % 360
            if currentHeading < 0: currentHeading += 360

        # rest of the animation code
        xs.append(currentPos[0])
        ys.append(currentPos[1])

        pose.set_data([currentPos[0]], [currentPos[1]])
        trajectory_line.set_data(xs, ys)

def main():
    global path1;
    path1 = [Waypoint(x, y) for x, y in path1]
    animationObj = Animation()
    animationObj.showAnimation()


main()
plt.close()

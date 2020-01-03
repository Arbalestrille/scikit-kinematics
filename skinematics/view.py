"""Visualization of IMU data

- An interactive viewer for time-series data ("view.ts")

- An animation of 3D orientations, expressed as quaternions
  ("view.orientation")

For the time-series viewer, variable types that can in principle be plotted
are:

* ``np.ndarray``
* ``pd.core.frame.DataFrame``
* ``pd.core.series.Series``

Viewer can be used to inspect a single variable, or to select one from the
current workspace.

Notable aspects
---------------

- Based on Tkinter, to ensure that it runs on all Python installations.
- Resizable window.
- Keyboard-based interaction.
- Logging of marked events.

.. autosummary::

   Orientation_OGL
   orientation
   ts

"""

import sys
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from os.path import expanduser, join
from mpl_toolkits.mplot3d import axes3d  # noqa:F401
import matplotlib.animation as animation
from skinematics import rotmat
from skinematics.vector import rotate_vector
# For Orientation_Viewers
# import pygame
# To avoid the annoying "Hello from the pygame community":
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import OpenGL.GL as gl          # noqa: E402
import OpenGL.GLU as glu        # noqa: E402

# List if plottable datatypes
_PLOTTABLE = [np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series]


class Orientation_OGL:
    """Orientation viewer utilizing OpenGL

    In the "zero" orientation, the pointer indicating the 3D orientation
    will point towards the lower right. In the display, the (x/y/z)-axes
    point in the (lower_right/lower_left/up) direction, respectively.

    Parameters
    ----------
    quats : (Nx3) or (Nx4) array
        Quaternions containing the orientation time series
    R_init : (3x3 array)
        Rotation matrix defining the initial orientation. Default is
        np.eye(3)
    win_width : integer
        Pixel-width of the display window.
    win_height : integer
        Pixel-height of the display window.
    win_title : string
        Title for the display window.

    Note
    ----
    The user is responsible for ensuring the input quaternions match the
    right-handed coordinate frame shown in the display.  This requires
    remapping and reorienting the vector part of the quaternions onto OGL's
    convention, whereby the x-axis points to the right of the window, the
    y-axis points up, and the z-axis poins towards the viewer (away from
    the screen).

    Examples
    --------
    >>> in_file = r'./tests/data/data_xsens.txt'
    >>> from skinematics.sensors.xsens import XSens
    >>> data = XSens(in_file)
    >>> viewer = Orientation_OGL(quat_in=data.quat)
    >>> viewer.run(looping=False, rate=100)

    """

    def __init__(self, quats=None, R_init=np.eye(3),
                 win_width=800, win_height=600, win_title=None):
        """Initialize the OpenGL-viewer"""

        # Initial orientation
        self.R_init = R_init

        # Camera
        self.cam_pos = [0.2, 0.2, 0]
        self.cam_target = [0, 0, -1]
        self.cam_up = [0, 1, 0]

        # OpenGL to Skinematics (OGL_x=skin_x; OGL_y=skin_z; OGL_z=-skin_y)
        x = [1, 0, 0]
        y = [0, 0, -1]
        z = [0, 1, 0]
        self.openGL2skin = np.column_stack((x, y, z))

        # Initialize the pygame graphics setup
        pygame.init()
        self.display = (win_width, win_height)
        pygame.display.set_mode(self.display,
                                pygame.DOUBLEBUF | pygame.OPENGL)
        if win_title:
            pygame.display.set_caption(win_title)

        self.define_elements()
        self.quats = quats

    def define_elements(self):
        """Define the visual components"""

        # Define the pointer
        delta = 0.01
        vertices = (            # on OGL x-y plane
            (0, -0.2, delta),   # top lower left
            (0, 0.2, delta),    # top upper left
            (0.6, 0, delta),    # top front
            (0, -0.2, -delta),  # bottom lower left
            (0, 0.2, -delta),   # bottom upper left
            (0.6, 0, -delta),   # bottom front
            )
        # Get quaternion of initial orientation in OGL CS
        q_init = rotmat.convert(self.R_init, to="quat")
        # Compute vertices in initial orientation using our convention
        self.vertices = (rotate_vector(vertices, q_init) @
                         self.openGL2skin.T)

        self.edges = (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5))

        self.colors = dict(
            pointer=((0.8, 0, 0),     # reddish (top)
                     (0.7, 0.7, 0.6),  # grayish (bottom)
                     (1, 1, 1)),       # white (edges)
            axes=((1, 0, 0),           # red (our x)
                  (0, 0, 1),           # blue (our z)
                  (0, 1, 0)))          # green (our y)

        self.surfaces = (
            (0, 1, 2),
            (3, 4, 5),
            (0, 1, 3))

        # Define the axes
        self.axes_endpts = np.array(
            [[-1, 0, 0],
             [1, 0, 0],
             [0, -1, 0],
             [0, 1, 0],
             [0, 0, -1],
             [0, 0, 1]])

        self.axes = ((0, 1), (2, 3), (4, 5))

    def draw_axes(self):
        """Draw the axes"""

        # gl.glLineWidth(2)
        gl.glBegin(gl.GL_LINES)

        for i, line in enumerate(self.axes):
            gl.glColor3fv(self.colors["axes"][i])
            for vertex in line:
                gl.glVertex3fv(self.axes_endpts[vertex])

        gl.glEnd()

    def draw_pointer(self, vertices):
        """Draw the triangle that indicates 3D orientation"""

        gl.glBegin(gl.GL_TRIANGLES)
        colors = self.colors["pointer"]
        for (color, surface) in zip(colors, self.surfaces):
            for vertex in surface:
                gl.glColor3fv(color)
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()

        gl.glBegin(gl.GL_LINES)
        gl.glColor3fv(colors[2])
        for edge in self.edges:
            for vertex in edge:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()

    def draw_text(self, pos, text, color):
        """Draw text at given position"""
        # Append alpha
        color.append(255)
        font = pygame.font.SysFont("Courier", 18, True)
        text_surface = font.render(text, True, color, (0, 0, 0, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        gl.glRasterPos3d(*pos)
        gl.glDrawPixels(text_surface.get_width(),
                        text_surface.get_height(),
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        text_data)

    def run(self, rate=100, looping=True):
        """Run the viewer

        Parameters
        ----------
        rate : integer
            Sample rate for the display [Hz]. Lower numbers result in
            slower display.
        looping : boolean
            If set to "True", the display will loop until the window is closed.

        """
        dt = int(1 / rate * 1000)  # [msec]

        # Camera properties, e.g. focal length etc
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        glu.gluPerspective(45, (self.display[0] / self.display[1]),
                           0.1, 50.0)
        gl.glTranslatef(0.0, 0.0, -3)

        counter = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            counter = np.mod(counter, self.quats.shape[0])
            if not looping and (counter + 1) == (self.quats.shape[0] - 1):
                break

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_DEPTH_TEST)

            # Camera position
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            glu.gluLookAt(
                self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                self.cam_target[0], self.cam_target[1], self.cam_target[2],
                self.cam_up[0], self.cam_up[1], self.cam_up[2])

            # Scene elements
            gl.glPushMatrix()
            vertices_i = rotate_vector(self.vertices, self.quats[counter])
            self.draw_pointer(vertices_i)
            gl.glPopMatrix()
            self.draw_axes()
            # Label *our* x coordinate
            self.draw_text((1, 0, 0), r"x (+)",
                           [x * 255 for x in self.colors["axes"][0]])
            # Label *our* y coordinate
            self.draw_text((0, 0, -1), r"y (+)",
                           [x * 255 for x in self.colors["axes"][2]])
            # Label *our* z coordinate
            self.draw_text((0, 1, 0), r"z (+)",
                           [x * 255 for x in self.colors["axes"][1]])
            # Draw counter
            self.draw_text((1, -1, 0), r"Index: {}".format(counter),
                           [x * 255 for x in self.colors["pointer"][2]])
            counter += 1

            pygame.display.flip()
            pygame.time.wait(dt)


def orientation(quats, out_file=None, title_text=None, deltaT=100):
    """Visualization of 3D orientations as animated triangle

    Calculate orientation of an arrow-patch to visualize a quaternion.
    Uses `_update_func` for the display.

    .. figure:: .static/orientation_viewer.png
       :scale: 25%

    Parameters
    ----------
    quats : array [(N,3) or (N,4)]
        Quaterions describing the orientation.
    out_file : string
        Path- and file-name of the animated out-file
        (".mp4"). [Default=None]
    title_text : string
            Name of title of animation [Default=None]
    deltaT : int
            interval between frames [msec]. Smaller numbers make faster
            animations.

    Example
    -------
        To visualize a rotation about the (vertical) z-axis:

    >>> # Set the parameters
    >>> omega = np.r_[0, 10, 10]     # [deg/s]
    >>> duration = 2
    >>> rate = 100
    >>> q0 = [1, 0, 0, 0]
    >>> out_file = 'demo_patch.mp4'
    >>> title_text = 'Rotation Demo'
    >>>
    >>> # Calculate the orientation
    >>> num_rep = duration*rate
    >>> omegas = np.tile(omega, [num_rep, 1])
    >>> q = skin.quat.calc_quat(omegas, q0, rate, 'sf')
    >>>
    >>> orientation(q, out_file, 'Well done!', deltaT=1000./rate)

    Note
    ----
    Seems to be slow. So unless you need a movie, better use "Orientation_OGL".

    """

    # Initialize the 3D-figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the arrow-shape and the top/bottom colors
    delta = 0.01    # "Thickness" of arrow
    corners = [[0, 0, 0.6],
               [0.2, -0.2, 0],
               [0, 0, 0]]
    colors = ['r', 'b']

    # Calculate the arrow corners
    corner_array = np.column_stack(corners)

    corner_arrays = []
    corner_arrays.append(corner_array + np.r_[0., 0., delta])
    corner_arrays.append(corner_array - np.r_[0., 0., delta])

    # Calculate the new orientations, given the quaternion orientation
    all_corners = []
    for quat in quats:
        all_corners.append([rotate_vector(corner_arrays[0], quat),
                            rotate_vector(corner_arrays[1], quat)])

    # Animate the whole thing, using 'update_func'
    num_frames = len(quats)
    ani = animation.FuncAnimation(fig, _update_func, num_frames,
                                  fargs=[all_corners, colors, ax, title_text],
                                  interval=deltaT)

    # If requested, save the animation to a file
    if out_file is not None:
        try:
            ani.save(out_file)
            print('Animation saved to {0}'.format(out_file))
        except ValueError:
            print('Sorry, no animation saved!')
            print('You probably have to install "ffmpeg", and add it to PATH.')

    plt.show()


def _update_func(num, all_corners, colors, ax, title=None):
    """For 3D plots it seems to be impossible to only re-set the data values,
    so the plot has to be cleared and re-generated for each frame
    """

    # Clear previous plot
    ax.clear()

    # Plot coordinate axes
    ax.plot([-1, 1], [0, 0], [0, 0])
    ax.plot([0, 0], [-1, 1], [0, 0])
    ax.plot([0, 0], [0, 0], [-1, 1])

    # Format the plot
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        # Plot and color the top- and bottom-arrow
        for up_down in range(2):
            corners = all_corners[num][up_down]
            ph = ax.plot_trisurf(corners[:, 0], corners[:, 1], corners[:, 2])
            ph.set_color(colors[up_down])

        if title is not None:
            plt.title(title)

    except RuntimeError:
        # When the triangle is exactly edge-on "plot_trisurf" seems to have
        # a numerical problem
        print('Cannot show triangle edge-on!')
    return


class Display:
    def __init__(self, master, data=None):
        """Create all frames, buttons and labels"""

        if type(data) not in _PLOTTABLE:
            self.inDictionary = data
            data = None

        self.master = master

        if data is None:
            data = np.arange(1)

        if data.ndim == 1:
            data = np.atleast_2d(data).T
        self.numData = data.shape[1]

        # Generate the figure -------------------------------------------
        fig, self.axs = plt.subplots(nrows=self.numData, sharex=True,
                                     sharey=False)

        if self.numData == 1:
            self.axs = [self.axs]

        self.lines = []
        self.rects = []
        self.zeros = []
        for ii in range(self.numData):
            self.lines.append(self.axs[ii].plot(data[:, ii]))

            # Zero line
            self.zeros.append(self.axs[ii].hlines(0, 0, len(data),
                                                  linestyle='dotted'))

            # Zoom box
            self.epsilon = 5
            (x0, x1, y0, y1) = (0, 0, 0, 0)
            self.rects.append(Line2D([x0, x1, x1, x0, x0],
                                     [y0, y0, y1, y1, y0],
                                     linestyle='dotted'))
            self.axs[ii].add_line(self.rects[-1])

        # Create the canvas
        self.canvas = FigureCanvasTkAgg(fig, master=master)
        # self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        # Keyboard and mouse control
        self.button = False
        self.marks = []
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        self.canvas.mpl_connect('motion_notify_event', self.onmotion)

        # Create and pack the widgets
        self.createWidgets()
        self.showAll()

        if 'inDictionary' in dir(self):
            self.selectPlotVar()

    def createWidgets(self):
        """Create frames, buttons, text, etc."""

        # Frame for the slider
        frame_0_top = tk.Frame(self.master)

        self.scale = tk.Scale(frame_0_top, from_=0, to=1, resolution=0.01,
                              orient=tk.HORIZONTAL, length=500,
                              sliderlength=50, showvalue=False,
                              command=self.position)
        self.scale.set(0.0)
        self.scale.pack(fill=tk.X, ipadx=10)

        frame_0_bottom = tk.Frame(self.master)

        # Frame for Load/Exit buttons -----------------------
        frame_1 = tk.Frame(frame_0_bottom)
        # Create 2 buttons
        self.button_exit = tk.Button(frame_1, text="Exit",
                                     foreground='red',
                                     command=self.exit)
        self.button_exit.pack(side="right")
        self.button_load = tk.Button(frame_1, text="Load",
                                     fore='green',
                                     command=self.selectPlotVar)
        self.button_load.pack(side="right")
        self.button_showAll = tk.Button(frame_1, text="ShowAll",
                                        foreground='blue',
                                        command=self.showAll)
        self.button_showAll.pack(side="right")

        # Frame for navigation buttons ----------------------
        frame_2 = tk.Frame(frame_0_bottom)
        # Add navigation buttons
        self.button_ff = tk.Button(frame_2, text=">>",
                                   command=self.fforward)
        self.button_ff.pack(side="right")
        self.button_ff = tk.Button(frame_2, text=">",
                                   command=self.forward)
        self.button_ff.pack(side="right")
        self.button_ff = tk.Button(frame_2, text="<",
                                   command=self.backward)
        self.button_ff.pack(side="right")
        self.button_ff = tk.Button(frame_2, text="<<",
                                   command=self.fbackward)
        self.button_ff.pack(side="right")

        # Frame for entering text -----------------------------
        frame_3 = tk.Frame(frame_0_bottom)

        # Subframe for upper/lower limit
        frame_3_lim = tk.Frame(frame_3)

        # Subsubframe for upper limit
        frame_upper = tk.Frame(frame_3_lim)
        frame_lower = tk.Frame(frame_3_lim)

        label_upper = tk.Label(frame_upper, text="Upper Limit")
        label_upper.pack(side='left')
        label_lower = tk.Label(frame_lower, text="Lower Limit")
        label_lower.pack(side='left')

        self.text_upper = tk.Entry(frame_upper)
        self.text_upper.pack(side='right')
        self.text_lower = tk.Entry(frame_lower)
        self.text_lower.pack(side='right')

        # Subframe for rate
        frame_3_rate = tk.Frame(frame_3)

        label_rate = tk.Label(frame_3_rate, text='Rate')
        self.text_rate = tk.Entry(frame_3_rate)
        self.text_rate.insert(0, '1')

        # Checkbutton for loggin
        frame_3_log = tk.Frame(frame_3)

        label_log = tk.Label(frame_3_log, text='Log')
        self.chkVar = tk.IntVar()
        log_check = tk.Checkbutton(frame_3_log,
                                   variable=self.chkVar,
                                   command=self.log)

        # Pack the elements, and assign key bindings ---------------------
        frame_0_top.pack()
        frame_0_bottom.pack()
        frame_1.pack(side="right")
        frame_2.pack(side="right", expand=1)
        frame_3.pack(side="left", expand=1)

        frame_3_lim.pack(side="left", padx=5, expand=1)
        frame_upper.pack()
        frame_lower.pack()

        label_upper.pack(side='left')
        self.text_upper.bind('<Return>', self.setUpperLimit)
        self.text_upper.pack(side='left')

        label_lower.pack(side='left')
        self.text_lower.bind('<Return>', self.setLowerLimit)
        self.text_lower.pack(side='left')

        frame_3_rate.pack(side='left', padx=5, expand=1)
        label_rate.pack()
        self.text_rate.bind('<Return>', self.setRate)
        self.text_rate.pack()
        # self.text_rate.event_generate('<Return>')
        self.setRate('<Return>')

        frame_3_log.pack(side='left', padx=5, expand=1)
        label_log.pack()
        log_check.pack()

    def log(self):
        """Log right mouse clicks"""

        # print('logging is {0}'.format(self.chkVar.get()))
        if self.chkVar.get() == 1 and 'logFile' not in dir(self):
            home = expanduser('~')
            self.logFile = join(home, 'default.log')
            print('right-Mouse clicks are logged into {0}'.
                  format(self.logFile))

    def onmotion(self, event):
        """Event for mouse dragging"""

        if self.button:
            self._stop = (event.x, event.y, event.xdata, event.ydata)
            x = [self._start[2], self._stop[2]]
            y = [self._start[3], self._stop[3]]

            for ii in range(self.numData):
                self.rects[ii].set_xdata([x[0], x[1], x[1], x[0], x[0]])
                self.rects[ii].set_ydata([y[0], y[0], y[1], y[1], y[0]])
            # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                # event.button, event.x, event.y, event.xdata, event.ydata))

        self.canvas.draw()

    def onclick(self, event):
        """Select the button-down position"""

        if event.button == 1:
            # left mouse click
            self._start = (event.x, event.y, event.xdata, event.ydata)
            self.button = True
        if event.button == 3:
            # right mouse click
            if self.chkVar.get() == 1:
                # right mouse click
                self.marks.append(event.xdata)
                for ii in range(self.numData):
                    self.axs[ii].vlines(event.xdata, self.range[2],
                                        self.range[3])

        self.canvas.draw()

    def onrelease(self, event):
        """Select the button-up position, and zoom in on the selected range"""

        if self.button:         # only for left mouse clicks
            self._stop = (event.x, event.y, event.xdata, event.ydata)
            self.button = False

            # Only zoom in if a "reasonably" large area has been selected
            start = np.r_[self._start[:2]]
            stop = np.r_[self._stop[:2]]
            x, y = stop-start
            curDist = np.hypot(x, y)

            if curDist > self.epsilon:
                # Zoom in
                for ii in range(self.numData):
                    self.axs[ii].set_xlim([min(self._start[2], self._stop[2]),
                                           max(self._start[2], self._stop[2])])
                    self.axs[ii].set_ylim([min(self._start[3], self._stop[3]),
                                           max(self._start[3], self._stop[3])])

                xLim = self.axs[0].get_xlim()
                self.xRange = np.diff(xLim)[0]
                self.sliderMax = self.range[1]-self.xRange
                self.scale.set(xLim[0]/self.sliderMax)

                (x, y) = ([0, 0], [0, 0])
                for ii in range(self.numData):
                    self.rects[ii].set_xdata([x[0], x[1], x[1], x[0], x[0]])
                    self.rects[ii].set_ydata([y[0], y[0], y[1], y[1], y[0]])

                self.canvas.draw()

    def on_key_event(self, event):
        """Keyboard interaction"""

        # print('you pressed %s'%event.key)

        key = event.key

        # In Python 2.x, the key gets indicated as "alt+[key]"
        # Bypass this bug:
        if key.find('alt') == 0:
            key = key.split('+')[1]

        if key == 'f':
            self.forward()
        elif key == 'n':
            self.fforward()
        elif key == 'b':
            self.backward()
        elif key == 'p':
            self. fbackward()
        elif key == 'x':
            self.exit()
        elif key == 'a':
            self.showAll()
        elif key == 'z':
            self.zoom()

    def setRate(self, event):
        """Set the rate. Also use this to initialize a number of default values,
        and the values for the limit-boxes."""

        rate = float(self.text_rate.get())
        minVal = 0
        maxVal = 0
        for line in self.lines:
            x, y = line[0].get_data()
            pnts = np.arange(len(x))
            time = pnts/rate
            line[0].set_xdata(time)
            minVal = min(minVal, np.min(y))
            maxVal = max(maxVal, np.max(y))

        # Initially, show all data
        for ii in range(self.numData):
            # to avoid some spurious UserWarning
            max_range = np.max(time)
            if max_range == 0.0:
                max_range += 0.001
            self.axs[ii].set_xlim([0, max_range])

        # Make sure small numbers are nicely formatted
        if max(np.abs([minVal, maxVal])) < 0.01:
            strMin = '{0:.2e}'.format(minVal)
            strMax = '{0:.2e}'.format(maxVal)
        else:
            strMin = '{0:.2f}'.format(minVal)
            strMax = '{0:.2f}'.format(maxVal)

        self.text_lower.delete(0, tk.END)
        self.text_lower.insert(0, strMin)

        self.text_upper.delete(0, tk.END)
        self.text_upper.insert(0, strMax)

        # Set limit and range parameters
        curLim = self.axs[0].get_xlim()
        xMin = 0
        xMax = np.max(time)
        self.xRange, = np.diff(curLim)
        self.sliderMax = xMax-self.xRange
        self.range = [xMin, xMax, minVal, maxVal]

        # Draw canvas
        self.canvas.draw()
        self.canvas._tkcanvas.focus_set()

    def zoom(self):
        """Show all data on the y-axis, and 10% of all on x-axis"""

        for ii in range(self.numData):
            self.axs[ii].set_xlim([0, 0.1*self.range[1]])
            self.axs[ii].set_ylim(self.range[2:])
        self.xRange = 0.1*self.range[1]
        self.sliderMax = self.range[1]-self.xRange
        self.scale.set(0)
        self.canvas.draw()

    def showAll(self):
        """Show all data"""

        for ii in range(self.numData):
            # The "if" is to avoid some spurious "UserWarning"
            if np.diff(self.range[:2]) == 0.:
                self.axs[ii].set_xlim([-0.001, 0.001])
            else:
                self.axs[ii].set_xlim(self.range[:2])

            if np.diff(self.range[2:]) == 0.:
                self.axs[ii].set_ylim([-0.001, 0.001])
            else:
                self.axs[ii].set_ylim(self.range[2:])
        self.xRange = self.range[1]
        self.scale.set(0)
        self.canvas.draw()

    def setUpperLimit(self, event):
        """Set the `Upper Limit`"""

        UpperLimit = float(self.text_upper.get())
        self.text_lower.delete(0, tk.END)
        self.text_lower.insert(0, str(-UpperLimit))

        for ii in range(self.numData):
            self.axs[ii].set_ylim([-UpperLimit, UpperLimit])
        self.canvas.draw()
        self.canvas._tkcanvas.focus_set()

    def setLowerLimit(self, event):
        """Set the `Lower Limit`"""

        UpperLimit = float(self.text_upper.get())
        LowerLimit = float(self.text_lower.get())

        for ii in range(self.numData):
            self.axs[ii].set_ylim([LowerLimit, UpperLimit])
        self.canvas.draw()
        self.canvas._tkcanvas.focus_set()

    def update_xPos(self, xLim):
        """Set the x-range, and check that limits are within possible range"""

        xMin = self.range[0]
        xMax = self.range[1]
        # Check the minimum position
        if xLim[0] < xMin:
            xLim = [xMin, xMin+self.xRange]

        # Check the maximum position
        if xLim[1] > xMax:
            xLim = [xMax-self.xRange, xMax]

        # Update xlimits, and redraw the screen
        if self.sliderMax == 0:
            self.scale.set(0)
        else:
            self.scale.set(float(xLim[0])/self.sliderMax)

        for ii in range(self.numData):
            self.axs[ii].set_xlim(xLim)
        self.canvas.draw()

    def position(self, event):
        """Position window according to slider"""
        sliderPos = self.scale.get()
        newLim = np.r_[0, self.xRange] + sliderPos*self.sliderMax

        self.update_xPos(newLim)

    def forward(self):
        """Move data forward by half the visible distance"""
        for ii in range(self.numData):
            curLim = self.axs[ii].get_xlim()
        newLim = curLim + self.xRange/2

        self.update_xPos(newLim)

    def fforward(self):
        """Move data forward by one visible distance"""
        for ii in range(self.numData):
            curLim = self.axs[ii].get_xlim()
        newLim = curLim + self.xRange

        self.update_xPos(newLim)

    def backward(self):
        """Move data backward by half the visible distance"""
        for ii in range(self.numData):
            curLim = self.axs[ii].get_xlim()
        newLim = curLim - self.xRange/2

        self.update_xPos(newLim)

    def fbackward(self):
        """Move data back by one visible distance"""
        for ii in range(self.numData):
            curLim = self.axs[ii].get_xlim()
        newLim = curLim - self.xRange

        self.update_xPos(newLim)

    def exit(self):
        """Close the window, and if necessary save the right-clicked marks

        """

        if self.chkVar.get() == 1:
            np.savetxt(self.logFile, self.marks)
            print('right-Mouse clicks are saved into {0}'.format(self.logFile))

        self.master.quit()
        # If you don't use both, Python crashes under Python 2.x
        self.master.destroy()

    def updatePlot(self):
        """Update the figure"""

        for ii in range(self.numData):
            for line in self.lines[ii]:
                # Remove the old lines
                line.remove()
            self.zeros[ii].remove()

            # plot the new data
            # self.axs[ii].set_color_cycle(None)
            self.lines[ii] = self.axs[ii].plot(self.varValues)
            self.zeros[ii] = self.axs[ii].hlines(0, 0, len(self.varValues),
                                                 linestyle='dotted')
        self.master.title(self.varName)
        self.range = [0, len(self.varValues), np.min(self.varValues),
                      np.max(self.varValues)]
        self.showAll()

        self.text_upper.delete(0, tk.END)
        self.text_upper.insert(0, str(self.range[3]))
        self.text_lower.delete(0, tk.END)
        self.text_lower.insert(0, str(self.range[2]))

        self.canvas.draw()
        self.master.call('wm', 'attributes', '.', '-topmost', '1')

    def selectPlotVar(self):
        """Select a plottable variable from those in the workspace"""

        if 'inDictionary' not in dir(self):
            print('No additional variables available!')
        else:
            # Create a new window
            self.loadWindow = tk.Toplevel(self.master)
            varSelector = VarSelector(self.loadWindow, self)
            varSelector.master.title('Selection')


class VarSelector():
    """Class for the GUI-display of plottable items

    Analyze the current workspace for variables that can be plotted, and
    let the user select one.  Variable types that can in principle be
    plotted are:

        - np.ndarray
        - pd.core.frame.DataFrame
        - pd.core.series.Series

    """

    def __init__(self, selectionWindow,  mainApp):

        varList = mainApp.inDictionary.keys()
        plotList = []
        for curType in _PLOTTABLE:
            plotList += [var for var in varList if
                         type(mainApp.inDictionary[var]) == curType]

        self.master = selectionWindow
        self.frame = tk.Frame(selectionWindow)
        self.frame.grid()
        self.createWidgets(plotList)
        self.mainApp = mainApp

    def selectAndQuit(self):
        """Grab selected item, update main plot, and close VarSelector-GUI"""
        try:
            selected = self.items[int(self.listbox.curselection()[0])]
            self.mainApp.varName = selected
            self.mainApp.varValues = self.mainApp.inDictionary[selected]
            self.mainApp.updatePlot()

            home = expanduser('~')
            self.mainApp.logFile = join(home, selected + '.log')
            if self.mainApp.chkVar.get() == 1:
                print('right-Mouse clicks are logged into {0}'.
                      format(self.mainApp.logFile))

        except IndexError:
            # No selection made
            self.selected = ''
        self.master.destroy()

    def quitFun(self):
        """Quit VarSelector-GUI with no further action"""
        self.selected = ''
        self.master.destroy()

    def createWidgets(self, items):
        """Create the List, and the Quit-button for the VarSelector-GUI"""

        self.listbox = tk.Listbox(self.frame, name='varSelection',
                                  font=('times', 13))

        # Populate the list with the items provided
        self.items = items
        for item in items:
            self.listbox.insert(tk.END, item)

        # Place it on the grid
        self.listbox.grid(row=0, columnspan=2)

        # Create and place the Quit-button
        self.quitButton = tk.Button(self.frame, text='Select',
                                    command=self.selectAndQuit)
        self.quitButton.grid(row=1, column=0)
        self.quitButton = tk.Button(self.frame, text='Quit',
                                    command=self.quitFun)
        self.quitButton.grid(row=1, column=1)


def ts(data=None):
    """Interactive plot of time-series data (1D and 3D)

    In addition to the (obvious) GUI-interactions, the following options
    are available: Keyboard interaction:

        * f -> forward (+ 1/2 frame)
        * n -> next (+ 1 frame)
        * b -> back ( -1/2 frame)
        * p -> previous (-1 frame)
        * z -> zoom (x-frame = 10% of total length)
        * a -> all (adjust x- and y-limits)
        * x -> exit

    .. figure:: .static/viewer_ts3.png
       :scale: 50%

       Improved viewability of 3D data.

    .. figure:: .static/viewer_large.png
       :scale: 50%

       Interactively analyze time-series data

    Notes
    -----

    Often one wants to see data symmetrically about the zero-axis. To
    acilitate this display, adjusting the "Upper Limit" automatically sets
    the lower limit to the corresponding negative value.

    When "Log" is activated, right-mouse clicks are indicated with vertical
    bars, and the corresponding x-values are stored into the users
    home-directory, in the file "[varName].log". Since the name of the
    first value is unknown the first events are stored into "data.log".

    Pushing the "Load"-button shows you all the plottable variables in your
    namespace.  Plottable variables are:

        * ndarrays
        * Pandas DataFrames
        * Pandas Series

    Examples
    --------
    To view a single plottable variable:

    >>> x = np.random.randn(100,3)
    >>> view.ts(x)

    To select a plottable variable from the workspace:

    >>> x = np.random.randn(100,3)
    >>> t = np.arange(0,10,0.1)
    >>> y = np.sin(x)
    >>> view.ts(locals)

    """
    root = tk.Tk()
    Display(root, data)
    root.mainloop()


if __name__ == '__main__':
    import os.path as osp
    from sensors.xsens import XSens

    # Test OpenGL viewer
    in_file = osp.join(osp.dirname(__file__), "tests", "data",
                       "data_xsens.txt")
    data = XSens(in_file)
    viewer = Orientation_OGL(quats=data.quat)
    viewer.run(looping=False, rate=100)

    # 2D Viewer
    data = np.random.randn(100, 3)
    t = np.arange(0, 2 * np.pi, 0.1)
    x = np.sin(t)

    # Show the data
    ts(locals())
    # ts(data)
    print('Done')

    """
    # 3D Viewer ----------------
    # Set the parameters
    omega = np.r_[0, 10, 10]     # [deg/s]
    duration = 2
    rate = 100
    q0 = [1, 0, 0, 0]

    ## Calculate the orientation
    dt = 1./rate
    num_rep = duration*rate
    omegas = np.tile(omega, [num_rep, 1])
    q = quat.calc_quat(omegas, q0, rate, 'sf')

    # orientation(q)
    in_file = r'./tests/data/data_xsens.txt'
    from skinematics.sensors.xsens import XSens
    data = XSens(in_file)

    out_file = 'demo_patch.mp4'
    title_text = 'Rotation Demo'

    orientation(data.quat, out_file=None, title_text='Well done!')
    # Test pygame-viewer
    phi = np.arange(360)
    q = quat.deg2quat(np.column_stack((phi, np.zeros((len(phi), 2)))))

    viewer = Orientation_Viewer_pygame(quat_in=q)
    viewer.run()

    # Test OpenGL viewer:
    in_file = r'./tests/data/data_xsens.txt'
    from sensors.xsens import XSens
    data = XSens(in_file)
    orientation(data.quat, deltaT=5)
    viewer = Orientation_OGL(quat_in=data.quat)
    viewer.run(looping=False, rate=100)
    """

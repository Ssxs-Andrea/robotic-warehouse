"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

from gymnasium import error
import numpy as np
import six

from rware.warehouse import Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
# Add to color constants at the top
_OBSTACLE_COLOR = (128, 128, 128)  # Gray color for obstacles

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_GOAL_COLOR = (60, 60, 60)
# Add to color constants at the top
_CHARGING_STATION_COLOR = (100, 149, 237)  # Cornflower blue
_SHELF_PADDING = 2


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 30
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 2 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_goals(env)
        self._draw_shelfs(env)
        self._draw_obstacles(env)
        self._draw_charging_stations(env)  # Add this line
        self._draw_agents(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # HORIZONTAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # VERTICAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    def _draw_charging_stations(self, env):
        batch = pyglet.graphics.Batch()
        
        for station in env.charge_stations:
            x, y = station.x, station.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            
            # Draw charging station (lightning bolt symbol)
            glColor3ub(*_CHARGING_STATION_COLOR)
            
            # Draw a simple lightning bolt shape
            center_x = (self.grid_size + 1) * x + (self.grid_size + 1) / 2
            center_y = (self.grid_size + 1) * y + (self.grid_size + 1) / 2
            size = self.grid_size / 3
            
            glBegin(GL_TRIANGLES)
            # Top triangle
            glVertex2f(center_x, center_y + size)
            glVertex2f(center_x - size/2, center_y)
            glVertex2f(center_x + size/2, center_y)
            # Bottom triangle
            glVertex2f(center_x, center_y - size)
            glVertex2f(center_x - size/2, center_y)
            glVertex2f(center_x + size/2, center_y)
            glEnd()
            
            # Draw border
            glColor3ub(*_BLACK)
            glLineWidth(1)
            glBegin(GL_LINE_LOOP)
            glVertex2f(center_x, center_y + size)
            glVertex2f(center_x - size/2, center_y)
            glVertex2f(center_x, center_y - size)
            glVertex2f(center_x + size/2, center_y)
            glEnd()

    def _draw_obstacles(self, env):
        batch = pyglet.graphics.Batch()
        
        # Draw obstacles as solid rectangles
        for y in range(env.obstacles.shape[0]):
            for x in range(env.obstacles.shape[1]):
                if env.obstacles[y, x]:
                    # Convert to pyglet coordinates (y is inverted)
                    pyglet_y = self.rows - y - 1
                    
                    batch.add(
                        4,
                        gl.GL_QUADS,
                        None,
                        (
                            "v2f",
                            (
                                (self.grid_size + 1) * x + 1,  # TL - X
                                (self.grid_size + 1) * pyglet_y + 1,  # TL - Y
                                (self.grid_size + 1) * (x + 1),  # TR - X
                                (self.grid_size + 1) * pyglet_y + 1,  # TR - Y
                                (self.grid_size + 1) * (x + 1),  # BR - X
                                (self.grid_size + 1) * (pyglet_y + 1),  # BR - Y
                                (self.grid_size + 1) * x + 1,  # BL - X
                                (self.grid_size + 1) * (pyglet_y + 1),  # BL - Y
                            ),
                        ),
                        ("c3B", 4 * _OBSTACLE_COLOR),
                    )
        
        batch.draw()
        
        # Draw obstacle borders
        for y in range(env.obstacles.shape[0]):
            for x in range(env.obstacles.shape[1]):
                if env.obstacles[y, x]:
                    pyglet_y = self.rows - y - 1
                    
                    glColor3ub(*_BLACK)
                    glLineWidth(1)
                    glBegin(GL_LINE_LOOP)
                    glVertex2f((self.grid_size + 1) * x + 1,
                            (self.grid_size + 1) * pyglet_y + 1)
                    glVertex2f((self.grid_size + 1) * (x + 1),
                            (self.grid_size + 1) * pyglet_y + 1)
                    glVertex2f((self.grid_size + 1) * (x + 1),
                            (self.grid_size + 1) * (pyglet_y + 1))
                    glVertex2f((self.grid_size + 1) * x + 1,
                            (self.grid_size + 1) * (pyglet_y + 1))
                    glEnd()

    def _draw_shelfs(self, env):
        batch = pyglet.graphics.Batch()

        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            shelf_color = _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR

            # Draw the shelf (semi-transparent)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4ub(*shelf_color, 180)  # 180/255 alpha for transparency
            
            glBegin(GL_QUADS)
            glVertex2f((self.grid_size + 1) * x + _SHELF_PADDING + 1,
                    (self.grid_size + 1) * y + _SHELF_PADDING + 1)
            glVertex2f((self.grid_size + 1) * (x + 1) - _SHELF_PADDING,
                    (self.grid_size + 1) * y + _SHELF_PADDING + 1)
            glVertex2f((self.grid_size + 1) * (x + 1) - _SHELF_PADDING,
                    (self.grid_size + 1) * (y + 1) - _SHELF_PADDING)
            glVertex2f((self.grid_size + 1) * x + _SHELF_PADDING + 1,
                    (self.grid_size + 1) * (y + 1) - _SHELF_PADDING)
            glEnd()
            glDisable(GL_BLEND)

            # Draw weight number (only for shelves in request queue)
            if shelf in env.request_queue:
                label_x = x * (self.grid_size + 1) + (1/2) * (self.grid_size + 1)
                label_y = y * (self.grid_size + 1) + (1/2) * (self.grid_size + 1)
                
                weight_label = pyglet.text.Label(
                    str(shelf.weight),
                    font_name="Arial",
                    font_size=10,
                    bold=True,
                    x=label_x,
                    y=label_y,
                    anchor_x="center",
                    anchor_y="center",
                    color=(*_BLACK, 255),  # Plain black text
                )
                weight_label.draw()

        # Draw shelf borders
        for shelf in env.shelfs:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1
            
            glColor3ub(*_BLACK)
            glLineWidth(1)
            glBegin(GL_LINE_LOOP)
            glVertex2f((self.grid_size + 1) * x + _SHELF_PADDING + 1,
                    (self.grid_size + 1) * y + _SHELF_PADDING + 1)
            glVertex2f((self.grid_size + 1) * (x + 1) - _SHELF_PADDING,
                    (self.grid_size + 1) * y + _SHELF_PADDING + 1)
            glVertex2f((self.grid_size + 1) * (x + 1) - _SHELF_PADDING,
                    (self.grid_size + 1) * (y + 1) - _SHELF_PADDING)
            glVertex2f((self.grid_size + 1) * x + _SHELF_PADDING + 1,
                    (self.grid_size + 1) * (y + 1) - _SHELF_PADDING)
            glEnd()

    def _draw_goals(self, env):
        batch = pyglet.graphics.Batch()

        # draw goal boxes
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * _GOAL_COLOR),
            )
        batch.draw()

        # draw goal labels
        for goal in env.goals:
            x, y = goal
            y = self.rows - y - 1
            label_x = x * (self.grid_size + 1) + (1 / 2) * (self.grid_size + 1)
            label_y = (self.grid_size + 1) * y + (1 / 2) * (self.grid_size + 1)
            label = pyglet.text.Label(
                "G",
                font_name="Calibri",
                font_size=18,
                bold=False,
                x=label_x,
                y=label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_WHITE, 255),
            )
            label.draw()

    def _draw_agents(self, env):
        agents = []
        batch = pyglet.graphics.Batch()

        radius = self.grid_size / 3
        resolution = 6

        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            # Calculate battery percentage and color (green to red gradient)
            battery_pct = agent.battery_level / env.battery_capacity
            r = int(255 * (1 - battery_pct))
            g = int(255 * battery_pct)
            battery_color = (r, g, 0)
            
            # Draw battery outline (below agent)
            battery_width = self.grid_size / 2
            battery_height = self.grid_size / 10
            battery_x = (self.grid_size + 1) * col + (self.grid_size + 1 - battery_width) / 2
            battery_y = (self.grid_size + 1) * row + self.grid_size / 10
            
            # Draw battery outline
            glColor3ub(*_BLACK)
            glLineWidth(1)
            glBegin(GL_LINE_LOOP)
            glVertex2f(battery_x, battery_y)
            glVertex2f(battery_x + battery_width, battery_y)
            glVertex2f(battery_x + battery_width, battery_y + battery_height)
            glVertex2f(battery_x, battery_y + battery_height)
            glEnd()
            
            # Draw battery positive terminal
            terminal_width = battery_width / 6
            terminal_height = battery_height / 2
            glBegin(GL_QUADS)
            glVertex2f(battery_x + battery_width, battery_y + (battery_height - terminal_height)/2)
            glVertex2f(battery_x + battery_width + terminal_width, battery_y + (battery_height - terminal_height)/2)
            glVertex2f(battery_x + battery_width + terminal_width, battery_y + (battery_height + terminal_height)/2)
            glVertex2f(battery_x + battery_width, battery_y + (battery_height + terminal_height)/2)
            glEnd()
            
            # Draw battery level fill
            glColor3ub(*battery_color)
            fill_width = max(1, (battery_width - 2) * battery_pct)  # -2 for padding
            glBegin(GL_QUADS)
            glVertex2f(battery_x + 1, battery_y + 1)
            glVertex2f(battery_x + 1 + fill_width, battery_y + 1)
            glVertex2f(battery_x + 1 + fill_width, battery_y + battery_height - 1)
            glVertex2f(battery_x + 1, battery_y + battery_height - 1)
            glEnd()

            # Draw agent circle (change color if charging)
            if agent.is_charging:
                draw_color = _GREEN  # Green when charging
            else:
                draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR

            # make a circle for the agent
            verts = []
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                x = (
                    radius * math.cos(angle)
                    + (self.grid_size + 1) * col
                    + self.grid_size // 2
                    + 1
                )
                y = (
                    radius * math.sin(angle)
                    + (self.grid_size + 1) * row
                    + self.grid_size // 2
                    + 1
                )
                verts += [x, y]
            circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
            glColor3ub(*draw_color)
            circle.draw(GL_POLYGON)

            # Add numeric battery percentage label
            battery_label_x = col * (self.grid_size + 1) + (self.grid_size + 1) / 2
            battery_label_y = row * (self.grid_size + 1) + self.grid_size / 5 + battery_height
            
            battery_label = pyglet.text.Label(
                f"{int(agent.battery_level)}",
                font_name="Arial",
                font_size=8,
                bold=True,
                x=battery_label_x,
                y=battery_label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_BLACK, 255),
            )
            battery_label.draw()

            # Add capacity label above agent
            capacity_label_x = col * (self.grid_size + 1) + (self.grid_size + 1) / 2
            capacity_label_y = row * (self.grid_size + 1) + 3 * (self.grid_size + 1) / 4
            
            capacity_label = pyglet.text.Label(
                f"Cap:{agent.max_carry_weight}",
                font_name="Arial",
                font_size=8,
                bold=True,
                x=capacity_label_x,
                y=capacity_label_y,
                anchor_x="center",
                anchor_y="center",
                color=(*_BLACK, 255),
            )
            capacity_label.draw()

        # Draw agent directions
        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1,  # CENTER X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1,  # CENTER Y
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.RIGHT.value else 0
                        )  # DIR X
                        + (
                            -radius if agent.dir.value == Direction.LEFT.value else 0
                        ),  # DIR X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.UP.value else 0
                        )  # DIR Y
                        + (
                            -radius if agent.dir.value == Direction.DOWN.value else 0
                        ),  # DIR Y
                    ),
                ),
                ("c3B", (*_AGENT_DIR_COLOR, *_AGENT_DIR_COLOR)),
            )
        batch.draw()

    def _draw_badge(self, row, col, index):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(index),
            font_name="Times New Roman",
            font_size=9,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()
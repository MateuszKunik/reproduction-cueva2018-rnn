import numpy as np
from abc import ABC, abstractmethod


class RodentMotionModel:
    def __init__(self, seed=None, theta_sigma=0.1, p_move=0.1, v_max=0.2):
        self.rng = np.random.default_rng(seed)
        self.theta_sigma = theta_sigma
        self.p_move = p_move
        self.v_max = v_max
        self.reset()

    def reset(self, x=0.0, y=0.0, theta=0.0, speed=0.0):
        self.x = float(x)
        self.y = float(y)
        
        self.theta = float(theta) if theta else self.rng.uniform(-np.pi, np.pi)
        self.speed = float(speed)

    def _sample_theta(self):
        self.theta += self.rng.normal(0.0, self.theta_sigma)
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def _sample_speed(self):
        if self.rng.random() < self.p_move:
            self.speed = self.rng.beta(2,  1) * self.v_max
        else:
            self.speed = 0.0

    def _sample_speed_fast(self):
        self.speed = self.rng.beta(2,  1) * self.v_max

    def can_move(self):
      return self.rng.random() < self.p_move

    def propose_step(self):
        self._sample_theta()
        self._sample_speed()

        dx = self.speed * np.cos(self.theta)
        dy = self.speed * np.sin(self.theta)

        return dx, dy, self.theta, self.speed

    def propose_step_fast(self):
        self._sample_theta()
        self._sample_speed_fast()

        dx = self.speed * np.cos(self.theta)
        dy = self.speed * np.sin(self.theta)

        return dx, dy, self.theta, self.speed
    

class Arena(ABC): #TODO: not need abstracts?
    @abstractmethod
    def contains(self, x, y):
        pass

    def center(self):
        pass
    
    def clamp(self, x, y):
        raise NotImplementedError
    
    
class SquareArena(Arena):
    def __init__(self, size, eps=1e-6):
        if size <= 0:
            raise ValueError("Arena size must be positive.")
        
        self.size = float(size)
        self._half_size = size / 2.0
        self.eps = float(eps)

    def contains(self, x, y):
        # (x >= -hs) & (x <= hs) & (y >= -hs) & (y <= hs)
        return (
           ( (self._half_size + self.eps) >= x >= - (self._half_size + self.eps) ) & (
               (self._half_size + self.eps) >= y >= - (self._half_size + self.eps)
           )
        )
    
    def center(self):
        return 0.0, 0.0


class SimulationController:
    def __init__(self, rodent, arena, max_tries=100):
        self.rodent = rodent
        self.arena = arena
        self.max_tries = int(max_tries)

        self.reset()
    
    def reset(self):
        self.t = 0
        self.data = []

        x0, y0 = self.arena.center()
        self.rodent.reset(x=x0, y=y0)

    def _apply_step(self, x, y, theta, speed):
        self.t += 1
        self.rodent.x = x
        self.rodent.y = y
        self.rodent.theta = theta
        self.rodent.speed = speed

    def _record_step(self):
        record = {
                "t": self.t,
                "theta": self.rodent.theta,
                "speed": self.rodent.speed,
                "x": self.rodent.x,
                "y": self.rodent.y,
        }

        self.data.append(record)
        return record

    def step(self):
        x0, y0 = self.rodent.x, self.rodent.y

        for _ in range(self.max_tries):
            dx, dy, theta_candidate, speed_candidate = self.rodent.propose_step()
            x_candidate = x0 + dx
            y_candidate = y0 + dy

            if self.arena.contains(x_candidate, y_candidate):
                x1, y1 = x_candidate, y_candidate
                theta, speed = theta_candidate, speed_candidate
                break
        else:
            x1, y1 = x0, y0
            theta = self.rodent.theta
            speed = 0.0

        self._apply_step(x1, y1, theta, speed)

        return self._record_step()

    def step_fast(self):
        x0, y0 = self.rodent.x, self.rodent.y

        if self.rodent.can_move():
          for _ in range(self.max_tries):
            dx, dy, theta_candidate, speed_candidate = self.rodent.propose_step_fast()
            x_candidate = x0 + dx
            y_candidate = y0 + dy

            if self.arena.contains(x_candidate, y_candidate):
                x1, y1 = x_candidate, y_candidate
                theta, speed = theta_candidate, speed_candidate
                break

            x_candidate = x0 - dx #switching angle we always move away from the wall
            y_candidate = y0 + dy
            if self.arena.contains(x_candidate, y_candidate):
                x1, y1 = x_candidate, y_candidate
                theta_candidate = np.atan2(dy,-dx)
                theta, speed = theta_candidate, speed_candidate
                break

            x_candidate = x0 - dx
            y_candidate = y0 - dy
            if self.arena.contains(x_candidate, y_candidate):
                x1, y1 = x_candidate, y_candidate
                theta_candidate = np.atan2(-dy,-dx)
                theta, speed = theta_candidate, speed_candidate
                break

            x_candidate = x0 + dx
            y_candidate = y0 - dy
            if self.arena.contains(x_candidate, y_candidate):
                x1, y1 = x_candidate, y_candidate
                theta_candidate = np.atan2(-dy,dx)
                theta, speed = theta_candidate, speed_candidate
                break

        else:
            x1, y1 = x0, y0
            theta = self.rodent.theta
            speed = 0.0

        self._apply_step(x1, y1, theta, speed)

        return self._record_step()

    def generate(self, num_steps):
        self.reset()
        self._record_step()

        for _ in range(1, num_steps):
            self.step()
        
        return self.data

    def generate_fast(self, num_steps):
        self.reset()
        self._record_step()

        for _ in range(1, num_steps):
            self.step_fast()

        return self.data
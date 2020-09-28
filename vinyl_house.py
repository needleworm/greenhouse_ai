"""
    Author : Byunghyun Ban
    SBIE @ KAIST
    needleworm@kaist.ac.kr
    latest modification :
        2017.06.12.
"""


import numpy as np
import random

class Vinylhouse:
    current_temparature = 21 # Celcius Degree
    current_humidity_air = 70 # Percent
    current_humidity_ground = 50 # Percent
    current_plant_height = 0 # Percent
    is_human = False
    insect_detected = False
    pesticide_density = 0
    water_temparature = 13

    tic = 0
    fan_on_duration = 0
    curtain_open_duration = 0
    water_inject_inside_duration = 0
    water_inject_outside_duration = 0
    LED_on_duration = 0
    pesticide_duration = 0
    light_on_duration = 0
    nutrients_spray_duration = 0
    human_duration = 0

    allowed_minimum_temparature = 0
    allowed_maximum_temparature = 100
    allowed_minimum_humidity_air = 0
    allowed_maximum_humidity_air = 100
    allowed_minimum_humidity_ground = 0
    allowed_maximum_humidity_ground = 100
    plants_harvested_with_height = 100
    allowed_maximum_pesticide_density = 10

    outside_temparature = 18
    outside_humidity = 30
    outside_rain = False
    outside_sun = True

    parameter = []
    job_cueus = []

    harvested = False

    # Had better not to assign value on default constructor.
    def __init__(self, allowed_minimum_temparature, allowed_maximum_temparature,
                 allowed_minimum_humidity_air, allowed_maximum_humidity_air,
                 allowed_minimum_humidity_ground, allowed_maximum_humidity_ground,
                 plants_harvested_with_height, current_temparature=21,
                 current_humidity_air=70, current_humidity_ground=50, current_plant_height=0):

        self.allowed_minimum_temparature = allowed_minimum_temparature
        self.allowed_maximum_temparature = allowed_maximum_temparature
        self.allowed_minimum_humidity_air = allowed_minimum_humidity_air
        self.allowed_maximum_humidity_air = allowed_maximum_humidity_air
        self.allowed_minimum_humidity_ground = allowed_minimum_humidity_ground
        self.allowed_maximum_humidity_ground = allowed_maximum_humidity_ground
        self.plants_harvested_with_height = plants_harvested_with_height
        self.current_temparature = current_temparature  # Celcius Degree
        self.current_humidity_air = current_humidity_air  # Percent
        self.current_humidity_ground = current_humidity_ground  # Percent
        self.current_plant_height = current_plant_height  # Percent

        self.parameter = [allowed_minimum_temparature, allowed_maximum_temparature,
            allowed_minimum_humidity_air, allowed_maximum_humidity_air,
            allowed_minimum_humidity_ground, allowed_maximum_humidity_ground,
            plants_harvested_with_height, current_temparature,
            current_humidity_air, current_humidity_ground, current_plant_height]

    def reset(self):
        self.allowed_minimum_temparature = self.parameter[0]
        self.allowed_maximum_temparature = self.parameter[1]
        self.allowed_minimum_humidity_air =self.parameter[2]
        self.allowed_maximum_humidity_air = self.parameter[3]
        self.allowed_minimum_humidity_ground = self.parameter[4]
        self.allowed_maximum_humidity_ground = self.parameter[5]
        self.plants_harvested_with_height = self.parameter[6]
        self.current_temparature = self.parameter[7]  # Celcius Degree
        self.current_humidity_air = self.parameter[8]  # Percent
        self.current_humidity_ground = self.parameter[9]  # Percent
        self.current_plant_height = self.parameter[10]  # Percent
        self.tic = 0
        self.fan_on_duration = 0
        self.curtain_open_duration = 0
        self.water_inject_inside_duration = 0
        self.water_inject_outside_duration = 0
        self.LED_on_duration = 0
        self.pesticide_duration = 0
        self.light_on_duration = 0
        self.nutrients_spray_duration = 0
        self.human_duration = 0
        self.outside_temparature = 18
        self.outside_humidity = 30
        self.outside_rain = False
        self.outside_sun = True
        return self.observation()


    def observation(self):
        self._perturbation()
        observations = np.zeros((7), dtype=np.float32)
        observations[0] = self.current_temparature
        observations[1] = self.current_humidity_air
        observations[2] = self.current_humidity_ground
        observations[3] = self.current_plant_height
        observations[4] = self.pesticide_density
        observations[5] = float(self.is_human) * 100
        observations[6] = float(self.insect_detected) * 100

        operations = np.zeros((8), dtype=np.float32)

        if self.tic > 0:
            if self.fan_on_duration > 0:
                operations[0] = 1
                self.fan_on()
            if self.curtain_open_duration > 0:
                operations[1] = 1
                self.curtain_open
            if self.water_inject_inside_duration > 0:
                operations[2] = 1
                self.water_inject_inside()
            if self.water_inject_outside_duration > 0:
                operations[3] = 1
                self.water_inject_outside()
            if self.pesticide_duration > 0:
                operations[4] = 1
                self.pesticide_injection()
            if self.light_on_duration > 0:
                operations[5] = 1
                self.LED_on
            if self.nutrients_spray_duration > 0:
                operations[6] = 1
                self.nutrients_spray
            if self.human_duration > 0:
                operations[7] = 1

        self.tic += 1

        reward = 1

        if self.current_temparature < self.allowed_minimum_temparature:
            reward = -1
        if self.current_temparature > self.allowed_maximum_temparature:
            reward = -1
        if self.current_plant_height > self.plants_harvested_with_height + 10:
            reward = -1
        if self.current_humidity_air > self.allowed_maximum_humidity_air:
            reward = -1
        if self.current_humidity_air < self.allowed_minimum_humidity_air:
            reward = -1
        if self.current_humidity_ground < self.allowed_minimum_humidity_ground:
            reward = -1
        if self.current_humidity_ground > self.allowed_maximum_humidity_ground:
            reward = -1
        if self.is_human and self.pesticide_duration > 1:
            reward = -1
        if (self.harvested and self.current_plant_height > self.plants_harvested_with_height) and (self.current_plant_height < self.plants_harvested_with_height + 10):
            reward = 1000

        if reward != 1:
            Done = True
        else:
            Done = False

        return observations, operations, reward, Done

    def _perturbation(self):

        is_temprature_got_higher = None
        if self.tic % 2000 == 0 and self.tic is not 0:
            is_temprature_got_higher = False
            rand_temprature = np.random.randint(0, 40)

            if self.outside_temparature < rand_temprature:
                is_temprature_got_higher = True
            self.outside_temparature += gradient(self.outside_temparature, rand_temprature)
            if rand_temprature < 15:
                self.outside_rain = True
                self.outside_sun = False
            else:
                self.outside_rain = False
                self.outside_sun = True

        if self.outside_rain is True:
            self.outside_humidity = 100.0
            self.current_temparature += gradient(self.current_temparature, self.outside_temparature)
            self.current_humidity_air += gradient(self.current_humidity_air, self.outside_humidity)

        else:
            self.outside_humidity = np.random.randint(10, 30)
            self.current_temparature += gradient(self.current_temparature, self.outside_temparature) + 4
            if is_temprature_got_higher is True:
                self.current_humidity_air -= np.random.randint(0, 6)
            if self.current_plant_height > 50:
                self.current_humidity_air -= gradient(self.current_humidity_ground, self.outside_humidity)

        self.current_humidity_ground += gradient(self.current_humidity_ground, self.outside_humidity)
        self.current_plant_height += random.random() / 2

        if self.current_humidity_air > 100:
            self.current_humidity_air = 100.0
        if self.current_humidity_ground > 100:
            self.current_humidity_ground = 100.0

        rand = np.random.randint(0, 10)
        if self.is_human:
            self.human_duration = 30
            if rand < 8:
                self.is_human = False
        elif self.human_duration > 0:
            self.human_duration -= 1
        else:
            if rand < 7:
                self.is_human = True

        if self.pesticide_duration > 0:
            self.insect_detected = False
        else:
            if rand < 9:
                self.insect_detected = True
            else:
                self.insect_detected = False

    def fan_on(self, duration=10):

        if self.fan_on_duration > 0:
            self.current_temparature += gradient(self.current_temparature, self.outside_temparature)
            self.current_humidity_air += gradient(self.current_humidity_air, self.outside_humidity)
            self.pesticide_density *= 0.9
            self.current_humidity_ground += 0.1 * gradient(self.current_humidity_ground, self.outside_humidity)
            self.fan_on_duration -= 1
        else:
            self.fan_on_duration = duration

    def curtain_open(self, duration = 30):
        if self.curtain_open_duration > 0:
            self.current_temparature += 3 * gradient(self.current_temparature, self.outside_temparature)
            self.current_humidity_air += 3 * gradient(self.current_humidity_air, self.outside_humidity)
            self.pesticide_density *= 0.8
            self.current_humidity_ground += 0.5 * gradient(self.current_humidity_ground, self.current_humidity_air)
            self.curtain_open_duration -= 1
        else:
            self.curtain_open_duration = duration

    def water_inject_inside(self, duration = 10):

        if self.water_inject_inside_duration > 0:
            self.current_temparature += gradient(self.current_temparature, self.water_temparature)
            self.current_humidity_air *= 1.1
            self.pesticide_density *= 0.7
            self.current_humidity_ground *= 1.3
            self.water_inject_inside_duration -= 1
        else:
            self.water_inject_inside_duration = duration

    def water_inject_outside(self, duration=20):

        if self.water_inject_outside_duration > 0:
            self.current_temparature *= 0.5
            self.current_humidity_air *= 0.9
            self.current_humidity_ground *= 0.95
            self.water_inject_outside_duration -= 1
        else:
            self.water_inject_outside_duration = duration

    def pesticide_injection(self, duration = 10):

        if self.pesticide_duration > 0:
            self.current_humidity_air *= 1.1
            self.pesticide_density *= 2
            self.current_humidity_ground *= 1.2
            self.pesticide_duration -= 1
        else:
            self.pesticide_duration = duration

    def LED_on (self, duration = 50):

        if self.LED_on_duration > 0:
            self.current_temparature *= 1.01
            self.current_humidity_air *= 0.95
            self.current_humidity_ground *= 0.98
            self.LED_on_duration -= 1
        else:
            self.LED_on_duration = duration

    def nutrients_spray(self, duration=10):

        if self.nutrients_spray_duration > 0:
            self.current_humidity_air *= 1.1
            self.pesticide_density *= 0.8
            self.current_humidity_ground *= 1.2
            self.nutrients_spray_duration -= 1
        else:
            self.nutrients_spray_duration = duration

    def harvest(self):
        if self.current_plant_height > 10:
            self.harvested = True

    def apply_action(self, actions):
        action = np.argmax(actions)
        if action == 0:
            self.fan_on()
        if action == 1:
            self.curtain_open()
        if action == 2:
            self.water_inject_inside()
        if action == 3:
            self.water_inject_outside()
        if action == 4:
            self.pesticide_injection()
        if action == 5:
            self.LED_on
        if action == 6:
            self.nutrients_spray()
        if action == 7:
            self.harvest()

    def step(self, actions):
        self.apply_action(actions)
        return self.observation()

def gradient(inside, outside):
    
    gradient = (float(outside) - float(inside))
    decay = random.uniform(0.01, 0.2)
    return gradient * decay

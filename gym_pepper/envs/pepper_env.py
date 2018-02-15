import sys
import numpy as np
import cv2

from naoqi import ALProxy
import qi
import almath

import time
import random 
import thread
import socket
from multiprocessing import Value,Queue

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import face_recognition

            
class PepperEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __naoqi_config(self):
        # setup basic configerations
        self.ip_addr = '130.238.17.115'
        self.port_num = 9559
        self.session = qi.Session()
        try:
            self.session.connect("tcp://" + self.ip_addr + ":" + str(self.port_num))
        except RuntimeError:
            print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
                   "Please check your script arguments. Run with -h option for help.")

    def __video_config(self):
        # create video interface
        self.video_device = ALProxy('ALVideoDevice', self.ip_addr, self.port_num)

        # subscribe top camera
        AL_kTopCamera = 0
        AL_kQVGA = 1            # 320x240
        AL_kQQVGA = 0            # 160x120
        AL_kBGRColorSpace = 13
        AL_kYUV422 = 9
        try:
            self.capture_device = self.video_device.unsubscribe()
        except:
            pass
        finally:
            self.capture_device = self.video_device.subscribeCamera(
                "pepper_top_camera_%s"%time.time(), AL_kTopCamera, AL_kQQVGA, AL_kBGRColorSpace, 30)
        self.screen_width = 160 # window configerations
        self.screen_height = 120
        self.image = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)

    def __gym_config(self):
        # set observation and action spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.action_space = spaces.Discrete(3) # Set with 8 elements {0, 1, 2, ..., 7}

    def __face_detection_config(self):
        # face recognizer configeration
        alex_image = face_recognition.load_image_file("experiments/people/people-alex.jpg")
        alex_face_encoding = face_recognition.face_encodings(alex_image)[0]
        
        self.known_face_encodings = [
            alex_face_encoding
        ]
        self.known_face_names = [
            "Alex",
        ]
    def __body_state_config(self):
        # disable autonomous life
        self.life = ALProxy('ALAutonomousLife', self.ip_addr, self.port_num)
        self.life.setAutonomousAbilityEnabled('All',False)
        
        self.awareness = self.session.service('ALBasicAwareness')
        self.awareness.setEnabled(False)

        self.leds = self.session.service('ALLeds')
        self.leds.setIntensity('AllLeds', 1.0)

        self.motion  = self.session.service("ALMotion")
        self.motion.setStiffnesses("Head", 1.0)
        names            = "HeadYaw"
        angles           = 0.0*almath.TO_RAD
        fractionMaxSpeed = 0.1
        self.motion.setAngles(names,angles,fractionMaxSpeed)

        names            = "HeadPitch"
        angles           = 0.0*almath.TO_RAD
        fractionMaxSpeed = 0.1
        self.motion.setAngles(names,angles,fractionMaxSpeed)

    def __init__(self):
        self.__naoqi_config()
        self.__video_config()
        self.__gym_config()
        self.__body_state_config()

        # remmeber the starting time
        self.start_time = time.time()

        # face detector configeration
        self.memory = ALProxy("ALMemory",self.ip_addr,self.port_num)
        self.session = self.memory.session()
        self.subscriber = self.session.service("ALMemory").subscriber("FaceDetected")
        self.subscriber.signal.connect(self.on_human_tracked)
        self.got_face = False
        self.face_detection = ALProxy("ALFaceDetection",self.ip_addr,self.port_num)
        self.face_detection.subscribe("HumanGreeter")

        self.__face_detection_config()

    def __del__(self):
        self.subscriber.unsubscribe("FaceDetected")
        self.face_detection.unsubscribe("HumanGreeter")
        self.capture_device = self.video_device.unsubscribe("pepper_top_camera")


    def on_human_tracked(self,value):
        if value == []:  # empty value when the face disappears
            self.got_face = False
        else:  # only speak the first time a face appears
            self.got_face = True
            
    def touch_sensor(self):
        if self.got_face:
            return 4
        else:
            return -1

    def wait(self):
        time.sleep(0.2)
        reward = 0.0001
        return reward


    def hello(self):
        time.sleep(0.2)
        if self.got_face:
            reward = 0.5
        else:
            reward = -1
        return reward

        names = list()
        times = list()
        keys = list()

        #names.append("LElbowRoll")
        #times.append([1, 1.5, 2, 2.5])
        #keys.append([-1.02102, -0.537561, -1.02102, -0.537561])

        #names.append("LElbowYaw")
        #times.append([1, 2.5])
        #keys.append([-0.66497, -0.66497])

        #names.append("LHand")
        #times.append([2.5])
        #keys.append([0.66])

        #names.append("LShoulderPitch")
        #times.append([1, 2.5])
        #keys.append([-0.707571, -0.707571])

        #names.append("LShoulderRoll")
        #times.append([1, 2.5])
        #keys.append([0.558505, 0.558505])

        #names.append("LWristYaw")
        #times.append([1, 2.5])
        #keys.append([-0.0191986, -0.0191986])
        #names2=["LElbowRoll","LElbowYaw","LHand","LShoulderPitch","LShoulderRoll","LWristYaw"]
        #angles=[-0.479966,-0.561996,0.66,1.30202,0.195477, -0.637045]
        #motion = ALProxy("ALMotion", self.ip_addr, self.port_num)
        #motion.setExternalCollisionProtectionEnabled("Arms", False)
        tts = ALProxy("ALTextToSpeech",self.ip_addr, self.port_num)
        tts.setParameter("speed", 100)
        tts.setLanguage("English")
        #motion.angleInterpolation(names, keys, times, True)
        tts.say("Hello")
        #motion.setAngles(names2,angles,0.3)
        if self.got_face:
            reward = 0.5
        else:
            reward = -1
        return reward

    def shake_hand(self):
        time.sleep(0.2)
        r=self.touch_sensor()
        if r>3:
            time.sleep(0.2)
        return r
        
        names = list()
        times = list()
        keys = list()
        r=0
        names.append("RHand")
        times.append([2])
        keys.append([0.98])



        names.append("RShoulderPitch")
        times.append([2])
        keys.append([-0.2058])


        names2=["RElbowRoll","RElbowYaw","RHand","RShoulderPitch","RShoulderRoll","RWristYaw"]
        angles=[0.479966,0.561996,0.66,1.30202,-0.195477, 0.637045]

        names3=["RHand"]
        angles2=[0.4]

        motion = ALProxy("ALMotion", self.ip_addr, self.port_num)
        motion.setExternalCollisionProtectionEnabled("Arms", False)
        tts = ALProxy("ALTextToSpeech",self.ip_addr, self.port_num)
        tts.setParameter("speed", 60)
        tts.setLanguage("English")
        motion.setExternalCollisionProtectionEnabled("Arms", False)
        motion.angleInterpolation(names, keys, times, True)
        r=self.touch_sensor()
        if r>3:
                tts.say("Nice to meet you")
                thread.start_new_thread(self.touch_sensor,())
                motion.setAngles(names3,angles2,0.4)
                time.sleep(2)
        motion.setAngles(names2,angles,0.4)
        return r

    def perform_actions(self,action):

#        print "The action is %d.\n1 and 2 are wait. 3 is hello and 4 is shake_hand."%action
        
        
        if action==0: #wait
            r=self.wait()

        elif action==1: #hello
            r=self.hello()

        elif action==2: #shake
            r=self.shake_hand()
        return r
    
    def _step(self, action):

        # Pre-process the information and store them in self.image

        result = self.video_device.getImageRemote(self.capture_device);
        if result == None:
            print 'cannot capture.'
        elif result[6] == None:
            print 'no image data string.'
        else:
            values = map(ord, list(result[6]))    # translate value to mat
            i = 0
            for y in range(0, self.screen_height):
                for x in range(0, self.screen_width):
                    self.image.itemset((y, x, 0), values[i + 0])
                    self.image.itemset((y, x, 1), values[i + 1])
                    self.image.itemset((y, x, 2), values[i + 2])
                    i += 3

        # Process the image in order to get face information and also display it
        
        rgb_small_frame = self.image
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(self.image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(self.image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('video', self.image)
        cv2.waitKey(1)
        

        
        ob = self.image
        reward = 0
        if (time.time() - self.start_time) > 0.3:
            episode_over = True
            self.start_time = time.time()
        else:
            episode_over = False
	reward = self.perform_actions(action)
        return ob, reward, episode_over, {}

    
    
    
    def _reset(self):
        result = self.video_device.getImageRemote(self.capture_device);
        if result == None:
            print 'cannot capture.'
        elif result[6] == None:
            print 'no image data string.'
        else:

            # translate value to mat
            values = map(ord, list(result[6]))
            i = 0
            for y in range(0, self.screen_height):
                for x in range(0, self.screen_width):
                    self.image.itemset((y, x, 0), values[i + 0])
                    self.image.itemset((y, x, 1), values[i + 1])
                    self.image.itemset((y, x, 2), values[i + 2])
                    i += 3
        
        state = self.image
        return state
    
    def _render(self, mode='human', close=False):
        pass
            
 


                




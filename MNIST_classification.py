__author__ = 'Dhyanesh Ghaghada'

# Importing Libraries
import pygame
from pygame.locals import *

import torch
import torch.nn as nn
from torchvision import transforms

import cv2

# Initializing PyGame.
pygame.init()

# MNIST model.
class MNIST_MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=5,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=5,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20*28*28, # Channels * Width * Height
                      out_features=10)
        )
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.classifier(x)
        return x

class MNIST_Classifier:
    '''
    This class is responsible for main classification using Deep Learning Model.
    '''
    def __init__(self, model, weights, transform=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # device agnostic code.
        
        self.model = model
        self.transform = transform
        self.model.load_state_dict(torch.load(weights)) # loading weights/model parameters.
        self.model = self.model.to(self.device)

        print("Using {} for classification.".format(self.device.upper()))
        self.classes = [
            '0 - Zero',
            '1 - One',
            '2 - Two',
            '3 - Three',
            '4 - Four',
            '5 - Five',
            '6 - Six',
            '7 - Seven',
            '8 - Eight',
            '9 - Nine',
        ] # All the classes.

    def convert_pixels_to_tensors(self, pixel_list):
        new_list = [] # list containing color value.
        for pixel in pixel_list:
            new_list.append(pixel.w_or_b)

        tensor_img = torch.Tensor(new_list)
        tensor_img = tensor_img.reshape(26, 26).T # reshaping from 784.
        tensor_img = torch.Tensor(cv2.resize(tensor_img.numpy(), (28, 28)))
        tensor_img = tensor_img.unsqueeze(dim=0).unsqueeze(dim=0) # unsqueezing for extra dim.
        if self.transform != None:
            tensor_img = self.transform(tensor_img)
        tensor_img = tensor_img.to(self.device) # Sending image to device.
        return tensor_img

    def classify(self, tensors):
        self.model.eval() # setting model mode to evaluation.
        with torch.inference_mode():
            y_pred = self.model(tensors)
            y_pred = y_pred.argmax(dim=1)
        return self.classes[y_pred.item()]
        

class Button:
    '''
    This class is used for creating buttons.
    '''
    def __init__(self, x, y, w, h):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.button = pygame.Rect(self.x, self.y, self.w, self.h) # creating button
        self.is_text = False
        self.is_clicked = False

    def is_pressed(self):
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            if self.button.collidepoint(mouse_pos):
                self.is_clicked = True
        else:
            self.is_clicked = False
        
        return self.is_clicked
    
    def create_text(self, size, text):
        self.is_text = True
        self.font = pygame.font.SysFont('Arial', size, True)
        self.text = self.font.render(text, False, (255, 255, 255))

    def render(self, screen):
        pygame.draw.rect(screen, (0,255,255), self.button)
        if self.is_text:
            screen.blit(self.text, (self.x + 5, self.y + 5))

class Pixel:
    '''
    This class is related to each pixel in a grid.
    '''
    def __init__(self, x, y, w, h):
        self.w_or_b = 0 # 0 means black and 1 means white.
        self.color = (0,0,0) # default black.
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.pixel = pygame.Rect(self.x, self.y, self.w, self.h) # creating a pixel.
    
    def render(self, screen):
        if self.w_or_b == 0:
            self.color = (0,0,0)
        elif self.w_or_b == 1:
            self.color = (255,255,255)

        pygame.draw.rect(screen, self.color, self.pixel)

class Grid:
    '''
    This class is responsible for all the pixels/grids.
    It is collection of pixels.
    '''
    def __init__(self, w, h, num_of_pixels):
        self.num_of_pixels = num_of_pixels
        self.grid_width = w // self.num_of_pixels
        self.grid_height = h // self.num_of_pixels
        self.pixel_list = [] # list with all pixels.

    def load_pixels(self):
        for i in range(self.num_of_pixels):
            for j in range(self.num_of_pixels):
                pixel_pos_x = i * self.grid_width
                pixel_pos_y = j * self.grid_height
                pixel = Pixel(pixel_pos_x, pixel_pos_y, self.grid_width, self.grid_height)
                self.pixel_list.append(pixel)

    def run(self, screen):
        for pixel in self.pixel_list:
            pixel.render(screen) # rendering pixel.
        
            # mouse handling.
            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                if pixel.pixel.collidepoint(mouse_pos):
                    pixel.w_or_b = 1

class Main:
    '''
    Main class.
    '''
    def __init__(self, w, h):
        # Settings
        self.WIDTH, self.HEIGHT = w, h
        self.running = True
        self.clock = pygame.time.Clock()

        self.grid = Grid(self.WIDTH, self.HEIGHT, 26) # creating a grid object.
        self.grid.load_pixels() # Loading all of the pixels.

        # This button is used to clear the screen if something is drawn.
        self.reset_button = Button(self.WIDTH/2-150, self.HEIGHT - 50, 90, 40)
        self.reset_button.create_text(30, 'Reset')

        # When the drawing of a number is complete, we procees further with classification using this button.
        self.classify_button = Button(self.WIDTH/2+75, self.HEIGHT - 50, 110, 40)
        self.classify_button.create_text(30, 'Classify')

        # model architecture.
        self.model = MNIST_MODEL()
        # transformations used in the image.
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-45,45))
        ])
        # model classification class
        self.classifier = MNIST_Classifier(self.model, 'MNIST_hand_digits_classification_model_1.pth', self.transform)
        self.result = None

        # screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('MNIST DIGIT CLASSIFICATION')
    
    def run(self, fps):
        while self.running:
            self.clock.tick(fps)
            self.screen.fill((0,0,0))

            self.grid.run(self.screen) # displaying a grid.
            
            if self.reset_button.is_pressed():
                # Setting all pixels to black
                for pixel in self.grid.pixel_list:
                    pixel.w_or_b = 0
                
                self.result = None

            # if 'classify' button is pressed then pass the image to model for classification.
            if self.classify_button.is_pressed():
                tensors = self.classifier.convert_pixels_to_tensors(
                    pixel_list=self.grid.pixel_list
                )
                self.result = self.classifier.classify(tensors=tensors)

            self.result_button = Button(0, 0, 300, 50)
            self.result_button.create_text(25, f'PREDICTION: {self.result}')
            self.result_button.render(self.screen)

            self.classify_button.render(self.screen)
            self.reset_button.render(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            pygame.display.update()
            pygame.display.flip()

# creating a main object of class Main.
main = Main(700,700)
main.run(120)
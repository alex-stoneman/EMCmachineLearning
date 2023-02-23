import pygame
import math
import random

# Set the size and parameters of the maze
RADIUS = 200
NUM_CIRCLES = 8
NUM_SPIRALS = 30
SPIRAL_GAP = 10
GAP_ANGLE = 20
OBSTACLE_PROBABILITY = 0.05

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = RADIUS * 2, RADIUS * 2
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Draw the outer circle of the maze
pygame.draw.circle(screen, (255, 255, 255), (RADIUS, RADIUS), RADIUS, 1)

# Draw the inner circles of the maze with gaps
for i in range(NUM_CIRCLES):
    r = RADIUS - ((i + 1) * RADIUS) // (NUM_CIRCLES + 1)
    gap_width = (2 * math.pi * r) / (NUM_CIRCLES + 1)
    gap_angle = math.degrees(gap_width / r)
    for j in range(NUM_CIRCLES - i):
        y = -r + (j + 0.5) * gap_width + gap_width
        circle = pygame.Surface((2 * r, 2 * r))
        circle.set_colorkey((0, 0, 0))
        circle.fill((0, 0, 0))
        pygame.draw.circle(circle, (255, 255, 255), (r, r), r, 1)
        pygame.draw.arc(circle, (255, 255, 255), (0, 0, 2 * r, 2 * r), gap_angle / 2, 360 - gap_angle / 2, 1)
        screen.blit(circle, (RADIUS - r, RADIUS - r - y))

# Draw the spirals of the maze
for i in range(NUM_SPIRALS):
    angle = i * (360 / NUM_SPIRALS)
    x, y = RADIUS, RADIUS
    for j in range(1000):
        x += math.cos(math.radians(angle)) * SPIRAL_GAP
        y += math.sin(math.radians(angle)) * SPIRAL_GAP
        if abs(math.hypot(x - RADIUS, y - RADIUS)) >= RADIUS:
            break
        if random.random() < OBSTACLE_PROBABILITY:
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 2)

        angle += GAP_ANGLE
        if angle >= 360:
            angle -= 360
        pygame.draw.line(screen, (255, 255, 255), (x, y), (x + math.cos(math.radians(angle)) * SPIRAL_GAP, y + math.sin(math.radians(angle)) * SPIRAL_GAP), 1)

# Update the display and wait for the user to close the window
pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

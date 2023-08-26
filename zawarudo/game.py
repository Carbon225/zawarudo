import pygame

from .zawarudo import ZaWarudo
from tqdm import tqdm


def main():
    dt = 5e-3
    steps = 6
    world = ZaWarudo()

    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    with tqdm() as pbar:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    exit()

            screen.fill((0, 0, 0))

            for (x, y), rgb in world.planets:
                pygame.draw.circle(screen, (rgb * 255).tolist(), (int(x * 20 + 400), int(y * 20 + 300)), 2)

            pygame.display.flip()

            world.step(dt, steps)
            pbar.update(1)

import numpy as np
import pygame
import sys
import multiprocessing
import queue

def ml_worker(input_queue, output_queue, model_path):
    """Runs in a separate process: loads TensorFlow model and performs predictions."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    while True:
        handwriting = input_queue.get()
        if handwriting is None:
            break
        pred = model.predict(
            [np.array(handwriting).reshape(1, 28, 28, 1)],
            verbose=0
        ).argmax()
        output_queue.put(int(pred))

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python recognition.py model")
    model_path = sys.argv[1]

    # Set up multiprocessing
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    ml_process = multiprocessing.Process(
        target=ml_worker, args=(input_queue, output_queue, model_path)
    )
    ml_process.start()

    # Start pygame
    pygame.init()
    size = width, height = 600, 400
    screen = pygame.display.set_mode(size)

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Fonts
    OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
    smallFont = pygame.font.Font(OPEN_SANS, 20)
    largeFont = pygame.font.Font(OPEN_SANS, 40)

    ROWS, COLS = 28, 28
    OFFSET = 20
    CELL_SIZE = 10

    handwriting = [[0] * COLS for _ in range(ROWS)]
    classification = None
    pending = False

    grid_rect = pygame.Rect(OFFSET, OFFSET, COLS * CELL_SIZE, ROWS * CELL_SIZE)

    # Static button rectangles (defined once)
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )

    try:
        while True:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if resetButton.collidepoint(event.pos):
                        handwriting = [[0] * COLS for _ in range(ROWS)]
                        classification = None
                        pending = False
                    elif classifyButton.collidepoint(event.pos):
                        input_queue.put([row[:] for row in handwriting])
                        pending = True

            screen.fill(BLACK)

            # Check for mouse press
            click, _, _ = pygame.mouse.get_pressed()
            if click == 1:
                mouse = pygame.mouse.get_pos()
            else:
                mouse = None

            # Draw each grid cell
            for i in range(ROWS):
                for j in range(COLS):
                    rect = pygame.Rect(
                        OFFSET + j * CELL_SIZE,
                        OFFSET + i * CELL_SIZE,
                        CELL_SIZE, CELL_SIZE
                    )

                    # If cell has been written on, darken cell
                    if handwriting[i][j]:
                        channel = 255 - (handwriting[i][j] * 255)
                        pygame.draw.rect(screen, (channel, channel, channel), rect)
                    # Draw blank cell
                    else:
                        pygame.draw.rect(screen, WHITE, rect)
                    pygame.draw.rect(screen, BLACK, rect, 1)

                    # If writing on this cell, fill in current cell and neighbors
                    if mouse and rect.collidepoint(mouse):
                        handwriting[i][j] = 250 / 255
                        if i + 1 < ROWS:
                            handwriting[i + 1][j] = 220 / 255
                        if j + 1 < COLS:
                            handwriting[i][j + 1] = 220 / 255
                        if i + 1 < ROWS and j + 1 < COLS:
                            handwriting[i + 1][j + 1] = 190 / 255
                        classification = None
                        pending = False

            # Draw Reset button
            resetText = smallFont.render("Reset", True, BLACK)
            resetTextRect = resetText.get_rect()
            resetTextRect.center = resetButton.center
            pygame.draw.rect(screen, WHITE, resetButton)
            screen.blit(resetText, resetTextRect)

            # Draw Classify button
            classifyText = smallFont.render("Classify", True, BLACK)
            classifyTextRect = classifyText.get_rect()
            classifyTextRect.center = classifyButton.center
            pygame.draw.rect(screen, WHITE, classifyButton)
            screen.blit(classifyText, classifyTextRect)

            # Check for classification result from the worker process
            try:
                classification = output_queue.get_nowait()
                pending = False
            except queue.Empty:
                pass

            # Show classification if one exists
            grid_size = OFFSET * 2 + CELL_SIZE * COLS
            if classification is not None:
                classificationText = largeFont.render(str(classification), True, WHITE)
                classificationRect = classificationText.get_rect()
                classificationRect.center = (
                    grid_size + ((width - grid_size) / 2),
                    100
                )
                screen.blit(classificationText, classificationRect)
            elif pending:
                pendingText = largeFont.render("...", True, WHITE)
                pendingRect = pendingText.get_rect()
                pendingRect.center = (
                    grid_size + ((width - grid_size) / 2),
                    100
                )
                screen.blit(pendingText, pendingRect)

            pygame.display.flip()

    finally:
        # Clean up the worker process
        input_queue.put(None)
        ml_process.join(timeout=2)
        if ml_process.is_alive():
            ml_process.terminate()
        pygame.quit()

if __name__ == '__main__':
    # 'spawn' prevents the child from inheriting Pygame's OpenGL state,
    # which is what caused the LLVM symbol collision / segfault.
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # already set
    main()

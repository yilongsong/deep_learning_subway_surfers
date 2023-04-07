

# datacollector.py

# yilong song
# Apr 6, 2023

# Run this script with python3 datacollector.py before each session of data collection
# Press escape key to exit
# Need to clean up images collected in the beginning and near the end

from pynput import keyboard # For taking screenshots upon keypresses
from pynput.keyboard import Key
import threading # To handle threading
import time
import random

import mss
import mss.tools

number_screenshots_per_second = 0.5

def screenshot(directory):
    with mss.mss() as sct:
        # Screenshot
        img = sct.grab({'top': 150, 'left': 0, 'width': 454, 'height': 600}) # This is the correct region to capture on my mac
        # When moving the BlueStacks simulator window to the top left corner of the screen.

        # Save file
        mss.tools.to_png(img.rgb, img.size, output='dataset/raw/'+directory+'/'+str(random.randint(100000, 1000000))+'.png')
    

def on_press(key):
    if key == Key.right:
        screenshot('right')
        print('right')
    elif key == Key.left:
        screenshot('left')
        print('left')
    elif key == Key.up:
        screenshot('up')
        print('up')
    elif key == Key.down:
        screenshot('down')
        print('down')
    elif key == Key.esc: # To exit
        print("EXIT")
        exit()

def start_key_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():

    key_listener_thread = threading.Thread(target=start_key_listener)
    key_listener_thread.start()

    while True:
        screenshot('noop')
        print('noop')
        time.sleep(1/number_screenshots_per_second)
            


if __name__ == '__main__':
    main()
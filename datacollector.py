

# datacollector.py

# yilong song
# Apr 6, 2023

# Run this script with python3 datacollector.py before each session of data collection
# Press escape key to exit
# Need to clean up images collected in the beginning and near the end

import pyautogui # For taking screenshots
import time # For taking screenshots at an interval (IS THIS NECESSARY?)
from pynput import keyboard # For taking screenshots upon keypresses
from pynput.keyboard import Key


file_name = 0

def screenshot(directory):
    global file_name
    img = pyautogui.screenshot(region=(0,80,908,1616)) # This is the correct region to capture on my mac
    # When moving the BlueStacks simulator window to the top left corner of the screen.
    img.save('dataset/'+directory+'/'+str(file_name)+'.png')
    file_name += 1
    

def on_press(key):
    if key == Key.right:
        screenshot('right')
    elif key == Key.left:
        screenshot('left')
    elif key == Key.up:
        screenshot('up')
    elif key == Key.down:
        screenshot('down')
    elif key == Key.esc: # To exit
        exit()

def main():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join() # Creating screenshot on separate thread


if __name__ == '__main__':
    main()
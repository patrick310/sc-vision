#!/bin/bash

import threading
import picamera
import time
import datetime
import sys, argparse, os, glob
import configs
import RPi.GPIO as GPIO

sys.setrecursionlimit(1000000)

resolution = (3280, 2464)
cam = picamera.PiCamera()
cam.resolution = resolution

duration = 2.5
saveLoc = '/home/pi/Pictures/'
photoLimit = None
timeLimit = None
imageName = 'image'


startTime = datetime.datetime.now()
photosTaken = 0

deleteThreadDone = False

TAB = '\t'
NEWL = '\n'

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(configs.light_ring_pin, GPIO.OUT)
GPIO.output(configs.light_ring_pin, GPIO.LOW)
def gatherArguments():
    print(NEWL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration')
    parser.add_argument('-s', '--saveLocation')
    parser.add_argument('-p', '--photoLimit')
    parser.add_argument('-t', '--timeLimit')
    parser.add_argument('-n', '--imageName')
    args = parser.parse_args()

    global duration
    if args.duration is None:
        print(TAB + 'Duration not defined. Duration defaulting to 5 seconds')
        duration = 5
    else:
        try:
            duration = int(args.duration)
            print(TAB + 'Duration defined to: ' + str(duration))
        except:
            print('ERR: Invalid duration given.')
            raise

    print(NEWL)

    global saveLoc
    if args.saveLocation is None:
        print(TAB + 'Save Location not defined. Save Location defaulting to:')
        print(TAB + ' -> ' + saveLoc)
    else:
        try:
            saveLoc = str(args.saveLocation)
            print(TAB + 'Save Location defined to:')
            print(TAB + ' -> ' + saveLoc)
        except:
            print('ERR: Invalid Save Location given.')
            raise

    print(NEWL)

    global photoLimit
    if args.photoLimit is None:
        print(TAB + 'Photo Limit not defined. Photo Limit set to None')
        photoLimit = None
    else:
        try:
            photoLimit = int(args.photoLimit)
            print(TAB + 'Photo Limit defined to: ' + str(photoLimit))
        except:
            print('ERR: Invalid Photo Limit given.')
            raise

    print(NEWL)

    global timeLimit
    if args.timeLimit is None:
        print(TAB + 'Time Limit not defined. Time Limit set to None')
    else:
        try:
            timeLimit = int(args.timeLimit)
            print(TAB + 'Time Limit defined to: ' + str(timeLimit) + ' hours')
        except:
            print('ERR: Invalid Time Limit given.')
            raise

    print(NEWL)

    if photoLimit is None and timeLimit is None:
        print(TAB + 'No Photo Limit or Time Limit defined. Use Cntrl-C to stop program.')
        print(NEWL)

    global imageName
    if args.imageName is None:
        print(TAB + 'Image Name not defined. Image Name defaulting to: ' + imageName)
    else:
        try:
            imageName = str(args.imageName)
            print(TAB + 'Image Name defined to: ' + imageName)
        except:
            print('ERR: Invalid Image Name given.')
            raise

    print(NEWL)



def stopCondition():
    stop = False

    global photoLimit
    if photoLimit is not None:
        global photosTaken
        if(photoLimit == photosTaken):
            return True, 'photoLimit'

    global timeLimit
    if timeLimit is not None:
        global startTime
        currentTime = datetime.datetime.now() - startTime
        if((currentTime.seconds / 3600) >= timeLimit):
            return True, 'timeLimit'

    return stop, ''

def nextName():
    counter = 0
    global imageName
    os.chdir(saveLoc)
    usedNames = list(glob.glob('*'))
    while True:
        name = imageName + str(counter) + '.jpg'
        if name not in usedNames:
            return name
        else:
            counter += 1
            continue

def cameraLoop():
    stop, condition = stopCondition()

    if(stop):
        return condition

    name = nextName()
    name = saveLoc + '/' + name

    global cam
    try:
        global duration
        time.sleep(duration)
        print(TAB + 'Capturing ' + str(name))
        GPIO.output(configs.light_ring_pin, GPIO.HIGH)
        cam.capture(name)
        GPIO.output(configs.light_ring_pin, GPIO.LOW)
        global photosTaken
        photosTaken += 1
    except:
        raise

    try:
        return cameraLoop()
    except:
        raise


def main():
    print('Starting AcquireImages.py program at ' + str(startTime))

    print('Evaluating arguments:')
    gatherArguments()
    print('Arguments evaluated')

    print('Starting camera loop ...')
    try:
        condition = cameraLoop()
        if(condition is 'photoLimit'):
            print('Stop condition met with condition: Photo Limit')
        elif(condition is 'timeLimit'):
            print('Stop condition met with condition: Time Limit')
        else:
            print('Unknown stop condition met ...')
    except KeyboardInterrupt as err:
        print('Program stopped with Cntrl-C')
    except Exception as err:
        print(NEWL)
        print('ERR: Exception generated before stop condition met.')
        print(TAB + 'Photos taken: ' + str(photosTaken))
        currentTime = datetime.datetime.now()
        h = (currentTime - startTime).seconds / 3600
        m = ((currentTime - startTime).seconds - h * 3600) / 60
        s = ((currentTime - startTime).seconds - m * 60 - h * 3600)
        print(TAB + 'Time elapsed: ' + str(h) + ' hours ' + str(m) + ' minutes ' + str(s) + ' seconds.')
        print(NEWL)
        raise err
    finally:
        print('Ending camera loop ...')
        print(TAB + 'Photos taken: ' + str(photosTaken))
        currentTime = datetime.datetime.now()
        h = (currentTime - startTime).seconds / 3600
        m = ((currentTime - startTime).seconds - h * 3600) / 60
        s = ((currentTime - startTime).seconds - m * 60 - h * 3600)
        print(TAB + 'Time elapsed: ' + str(h) + ' hours ' + str(m) + ' minutes ' + str(s) + ' seconds.')
        endTime = datetime.datetime.now()
        print('Exiting program at ' + str(endTime))
main()

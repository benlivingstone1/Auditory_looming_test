'''
***************************************************************************************************
                            AUDITORY LOOMING STIMULUS V1.0
    - This script runs the auditory looming task. The task plays a continuous baseline sound which 
        is interrupted with a series of 10 sounds that increase in amplitude over 0.4s. The program
        allows the user to calibrate the base and peak amplitudes on startup. 
    - This script also tracks the location of a chosen object, and whether the objects centroid is 
        in a user defined region. 
    - This script outputs a video of the tracked object and selected region, as well as a .csv file 
        of the centroid pixel coordinates. 

    - Ben Livingstone, April 2023
***************************************************************************************************
'''

import cv2 as cv
import numpy as np
import pyaudio
import multiprocessing
from multiprocessing import Process
import sys
import stimulus as stim


def point_inside(rect, point):
    in_roi = None
    test = cv.pointPolygonTest(np.array([(rect[0], rect[1]), 
                                (rect[0] + rect[2], rect[1]), 
                                (rect[0] + rect[2], rect[1] + rect[3]), 
                                (rect[0], rect[1] + rect[3])], dtype=np.float32), 
                                np.array((point[0], point[1]), dtype=np.float32), False)
    if test >= 0:
        in_roi = True
        color = (255,0,0)
    else:
        in_roi = False
        color = (0,255,0)
    
    return in_roi, color


def trigger_stim(play_stim, end_stream, wave):
    p = pyaudio.PyAudio()

    while True:
        play_stim.wait()
        stim_stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=44100,
                            output=True,
                            frames_per_buffer=1024)
        stim_stream.write(wave)
        stim_stream.close()

        play_stim.clear()

        if end_stream.is_set():
            stim_stream.close()
            p.terminate()
            break


def play_sound(end_stream, background):
    p = pyaudio.PyAudio()

    cont_stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True,
                    frames_per_buffer=1024)
    while not end_stream.is_set():
        cont_stream.write(background)
 
    cont_stream.stop_stream()
    cont_stream.close()
    p.terminate()
    return


# convert arg value to int if its a number
def if_int(value):
    try:
        out = int(value)
        return out
    except ValueError:
        return None


if __name__ == "__main__":
    # Show help
    if len(sys.argv) < 2:
        print(
            "Usage: object_tracker <video_name>\n"
            "examples:\n"
            "object_tracker ben/documents/video1.mp4"
        )
        sys.exit()

    # Set input video
    if if_int(sys.argv[1]) != None:
        sys.argv[1] = if_int(sys.argv[1])

    video = sys.argv[1]
    cap = cv.VideoCapture(video)

    # Exit if video is not opened
    if not cap.isOpened():
        print("Failed to open video.")
        sys.exit()

    # Get frame rate and size from the input video
    frameRate = cap.get(cv.CAP_PROP_FPS)
    frameSize = (
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )

    # Create a VideoWriter object and use size form input video
    # If you cannot open video, change fourcc to "MJPG"
    outputVid = cv.VideoWriter(
        "output.mp4", cv.VideoWriter_fourcc(*"MP4V"), frameRate, frameSize
    )

    # Check if the video writer was successfully created
    if not outputVid.isOpened():
        print("Error creating video writer")
        sys.exit()

    # Calibrate the MIN/MAX amplitude 
    fs = 44100
    base_amp = 0.2
    min_amp = stim.calibrate(base_amp)
    max_amp = stim.calibrate(0.7)

    # Create the constant amplitude background sound
    global background
    background = stim.noise_generator(min_amp, 50, 44100)
    hanning_window = int(fs * 0.5)
    hanning = np.hanning(hanning_window)
    rev_hanning = hanning[::-1]
    background[0:(hanning_window//2)] = background[0:hanning_window//2] * hanning[:(hanning_window//2)]
    background[-(hanning_window//2):] = background[-(hanning_window//2):] * rev_hanning[:(hanning_window//2)]

    # Create the stimulus wave
    # global wave
    wave = stim.wave_generator(min_amp, max_amp)
    # plt.plot(wave)
    # plt.show()

    # Initialize the tracker
    tracker = cv.TrackerKCF_create()

    # Perform the tracking process
    print("Starting the tracking process, pres ESC to quit.")

    # Create a csv file to save centroid location
    csv = open("centroid.csv", "w")

    # Declare an ROI if there isn't one already 
    roi = None
    while roi is None:
        ret, frame = cap.read()
        if not ret:
            break
        roi = cv.selectROI("tracker", frame, False)
        tracker.init(frame, roi)

    # Declare rectangle that will trigger stimulus 
    rect = None 

    while rect is None:
        ret, frame = cap.read()
        if not ret:
            break
        print("Please select the trigger region")
        rect = cv.selectROI("trigger", frame, False, fromCenter=False)
        
    cv.destroyAllWindows()

    # Create event that will trigger stimulus playback and end the stream 
    play_stim = multiprocessing.Event()
    end_stream = multiprocessing.Event()

    # Create threads that will handle the audio playback: 
    sound_thread = Process(target=play_sound, args=(end_stream, background))
    stim_thread = Process(target=trigger_stim, args=(play_stim, end_stream, wave))
    sound_thread.start()
    stim_thread.start()

    # Create a csv file to save centroid location
    csv = open("centroid.csv", "w")

    # Loop over all frames
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        cv.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[0]+rect[2]), int(rect[1]+rect[3])), (0, 255, 0), 2)

        # Update the tracking result
        success, roi = tracker.update(frame)

        if success:
            # Draw the tracked object
            center = (int(roi[0] + roi[2] / 2), int(roi[1] + roi[3] / 2))
            radius = 10  # circle radius
            in_roi, color = point_inside(rect, center)
            cv.circle(frame, center, radius, color, -1)  # draw green circle at center point

            # Add center to csv
            csv.write(str(center[0]) + "," + str(center[1]) + "," + str(in_roi) +"\n")

        if in_roi and in_roi != prev_state:
            play_stim.set()

        prev_state = in_roi

        # Show image with the tracked object
        cv.imshow("tracker", frame)

        # Write the frame to the output video
        outputVid.write(frame)

        # Quit on ESC button
        if cv.waitKey(1) == 27:
            end_stream.set()
            break

    # Release resources
    cap.release()
    outputVid.release()
    cv.destroyAllWindows()
    csv.close()
    
    # Close threads
    stim_thread.terminate()
    sound_thread.terminate()

    print("Finished tracking video")

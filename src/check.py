import cv2
import face_recognition
import glob

# This is a demo of running face recognition on live video from your webcam.
# It includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/2 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

known_face_encodings=[]
known_face_names=[]
path = '//home//ojas//PycharmProjects//Term_Project//src//Images//*.jpg'

files = glob.glob(path)
for imag in files:
    name=imag[57:-4]
    load_image = face_recognition.load_image_file(imag)
    known_face_encodings.append(face_recognition.face_encodings(load_image)[0])
    known_face_names.append(name)


## Load a sample picture and learn how to recognize it.
#ojas_image = face_recognition.load_image_file("ojas.JPG")
#ojas_face_encoding = face_recognition.face_encodings(ojas_image)[0]
#
## Load a second sample picture and learn how to recognize it.
#shaw_image = face_recognition.load_image_file("chaw.jpg")
#shaw_face_encoding = face_recognition.face_encodings(shaw_image)[0]
#
## Load a third sample picture and learn how to recognize it.
#sss_image = face_recognition.load_image_file("SSS.jpg")
#sss_face_encoding = face_recognition.face_encodings(sss_image)[0]
#
## Load a fourth sample picture and learn how to recognize it.
#achal_image = face_recognition.load_image_file("achal.jpg")
#achal_face_encoding = face_recognition.face_encodings(achal_image)[0]
#
## Load a fifth sample picture and learn how to recognize it.
#ekta_image = face_recognition.load_image_file("ekta.jpg")
#ekta_face_encoding = face_recognition.face_encodings(ekta_image)[0]
#
#
## Load a fifth sample picture and learn how to recognize it.
#yogesh_image = face_recognition.load_image_file("yogesh.jpg")
#yogesh_face_encoding = face_recognition.face_encodings(yogesh_image)[0]
#
#
## Create arrays of known face encodings and their names
#known_face_encodings = [
#    ojas_face_encoding,
#    shaw_face_encoding,
#    sss_face_encoding,
#    achal_face_encoding,
#    ekta_face_encoding,
#    yogesh_face_encoding
#]
#known_face_names = [
#    "Ojas Agarwal",
#    "Siddhant Shaw",
#    "Sourabh Shubham",
#    "Achal Gupta",
#    "Ekta Tyagi",
#    "Yogesh Sharma"
#]
#
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
unique=[]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            matches = list(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if min(matches)<=0.6:
                first_match_index = matches.index(min(matches))
                name = known_face_names[first_match_index]

            face_names.append(name)
    process_this_frame = not process_this_frame
    for name in face_names:
        if name not in unique:
            unique.append(name)
    count=len(unique)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/2 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (255, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (0, 255, 0), 1)

        #to show the count of people
        cv2.rectangle(frame, (0, 0), (30, 30), (255, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, str(count), (6, 24), font, 0.75, (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

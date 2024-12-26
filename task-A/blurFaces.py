import cv2
from main import prep_video

# Function to detect and blur faces in a video
def blur_faces(video_file_path, output_file_name):
    # Prepare the video and output file
    vid, out, total_frames = prep_video(video_file_path, output_file_name)
    
    if vid is None or out is None:
        print("Error: Video preparation failed.")
        return
    
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier("task-A/face_detector.xml")
    if face_cascade.empty():
        print("Error: Failed to load face detection model. Check file path.")
        return

    frame_count = 0
    print("Face blurring in progress...\n")
    
    # Read and process each frame
    while True:
        ret, frame = vid.read()
        if not ret:
            break  # Break if no frames are left

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Blur detected faces
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face  # Replace original face with blurred one

        # Write the processed frame to the output file
        out.write(frame)

    # Release video objects
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    print("Face blurring complete. Video released.\n")

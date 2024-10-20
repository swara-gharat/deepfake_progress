# import os
# import cv2
# import face_recognition
# import tempfile
# import streamlit as st
# from video_processing import detect_fake_video
# import matplotlib.pyplot as plt
# import numpy as np
# import subprocess
# # import ffmpeg

# # Video frame extraction function
# def frame_extract(path):
#     """Extract frames from a video file."""
#     vidObj = cv2.VideoCapture(path)
#     success = True
#     while success:
#         success, image = vidObj.read()
#         if success:
#             yield image

# # Function to create face-cropped videos
# def create_face_videos(file_path, out_dir):
#     """Process a video file and save cropped face videos."""
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
    
#     out_path = os.path.join(out_dir, "processed_video.mp4")
#     frames = []
#     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

    
#     for idx, frame in enumerate(frame_extract(file_path)):
#         if idx <= 150:
#             frames.append(frame)
#             if len(frames) == 4:
#                 all_faces = []
#                 for frm in frames:
#                     rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#                     face_locations = face_recognition.face_locations(rgb_frame)
#                     all_faces.extend(face_locations)
                
#                 if not all_faces:
#                     print(f"No faces detected in frames")
                
#                 for (top, right, bottom, left) in all_faces:
#                     for i in range(len(frames)):
#                         try:
#                             face_image = frames[i][top:bottom, left:right]
#                             if face_image.size == 0:
#                                 continue
#                             face_image = cv2.resize(face_image, (112, 112))
#                             out.write(face_image)
#                         except Exception as e:
#                             print(f"Error processing frame {i}: {e}")
#                 frames = []
    
#     out.release()
#     if os.path.exists(out_path):
#         print(f"Finished processing video: {out_path}")
#     else:
#         print(f"Failed to save processed video: {out_path}")
    
#     return out_path

# def extract_metadata(video_file):
#     """Extract metadata from a video file using ExifTool and print to terminal."""
#     try:
#         # Use full path to exiftool if it's not in PATH
#         exiftool_path = 'exiftool'  # Change this if exiftool is not in your system's PATH
#         result = subprocess.run([exiftool_path, '-a', '-u', '-g1', video_file], stdout=subprocess.PIPE)

#         # Decode the output to get metadata
#         metadata = result.stdout.decode('utf-8')
#         print("Extracted Metadata:\n", metadata)  # Print metadata to terminal
#         return metadata
#     except FileNotFoundError as e:
#         print(f"Failed to extract metadata: ExifTool not found. Ensure it is installed and in your PATH.")
#     except PermissionError as e:
#         print(f"Permission denied: {e}. Try running the script with elevated permissions.")
#     except Exception as e:
#         print(f"Failed to extract metadata: {str(e)}")
#     return None

# def main():
#     st.title("Deepfake Video Detection")

#     uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             video_path = tmp_file.name
        

#         st.write("Extracting metadata...")
#         extract_metadata(video_path)

#         processed_video_path = create_face_videos(video_path, 'Processed_Videos/')
#         st.write(f"Video path: {processed_video_path}")
    

# # Display the first frame
        
#         if st.button("Analyze Video"):
#             prediction = detect_fake_video(processed_video_path)
#             output = "REAL" if prediction[0] == 1 else "FAKE"
#             confidence = prediction[1]
#             st.write(f"Prediction: {output} with {confidence:.2f}% confidence")
            


      
# if __name__ == "__main__":
#     main()

import os
import cv2
import face_recognition
import tempfile
import streamlit as st
from video_processing import detect_fake_video
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# Video frame extraction function
def frame_extract(path):
    """Extract frames from a video file."""
    vidObj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image

# Function to create face-cropped videos
def create_face_videos(file_path, out_dir):
    """Process a video file and save cropped face videos."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(out_dir, "processed_video.mp4")
    frames = []
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

    for idx, frame in enumerate(frame_extract(file_path)):
        if idx <= 150:
            frames.append(frame)
            if len(frames) == 4:
                all_faces = []
                for frm in frames:
                    rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    all_faces.extend(face_locations)
                
                if not all_faces:
                    print(f"No faces detected in frames")
                
                for (top, right, bottom, left) in all_faces:
                    for i in range(len(frames)):
                        try:
                            face_image = frames[i][top:bottom, left:right]
                            if face_image.size == 0:
                                continue
                            face_image = cv2.resize(face_image, (112, 112))
                            out.write(face_image)
                        except Exception as e:
                            print(f"Error processing frame {i}: {e}")
                frames = []
    
    out.release()
    if os.path.exists(out_path):
        print(f"Finished processing video: {out_path}")
    else:
        print(f"Failed to save processed video: {out_path}")
    
    return out_path

def extract_metadata(video_file, metadata_dir):
    """Extract metadata from a video file using ExifTool and save it in a file."""
    try:
        # Ensure metadata directory exists
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)

        # Get video file name without extension
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        metadata_file_path = os.path.join(metadata_dir, f"{video_name}_metadata.txt")

        # Use full path to exiftool if it's not in PATH
        exiftool_path = 'exiftool'  # Change this if exiftool is not in your system's PATH
        result = subprocess.run([exiftool_path, '-a', '-u', '-g1', video_file], stdout=subprocess.PIPE)

        # Decode the output to get metadata
        metadata = result.stdout.decode('utf-8')

        # Save metadata to a file
        with open(metadata_file_path, 'w') as f:
            f.write(metadata)

        print(f"Metadata saved to: {metadata_file_path}")
        return metadata_file_path
    except FileNotFoundError as e:
        print(f"Failed to extract metadata: ExifTool not found. Ensure it is installed and in your PATH.")
    except PermissionError as e:
        print(f"Permission denied: {e}. Try running the script with elevated permissions.")
    except Exception as e:
        print(f"Failed to extract metadata: {str(e)}")
    return None

def main():
    st.title("Deepfake Video Detection")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Metadata extraction and saving
        st.write("Extracting metadata...")
        metadata_dir = "metadata"
        metadata_file = extract_metadata(video_path, metadata_dir)
        if metadata_file:
            st.write(f"Metadata saved to: {metadata_file}")

        processed_video_path = create_face_videos(video_path, 'Processed_Videos/')
        st.write(f"Video path: {processed_video_path}")
        
        if st.button("Analyze Video"):
            prediction = detect_fake_video(processed_video_path)
            output = "REAL" if prediction[0] == 1 else "FAKE"
            confidence = prediction[1]
            st.write(f"Prediction: {output} with {confidence:.2f}% confidence")

if __name__ == "__main__":
    main()

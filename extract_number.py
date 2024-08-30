# import cv2
# from ultralytics import YOLO
# import os
# import easyocr

# def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
#     model = YOLO(yolo_model_path)
#     reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader with English language support

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = r'D:\vChanel\spinning mill\gate_entrence1'
#     os.makedirs(output_folder, exist_ok=True)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     frame_index = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_index += 1

#         yolomodel = model(frame)

#         for output in yolomodel:
#             for detection in output.boxes:
#                 confi = detection.conf[0]
#                 class_name = model.names[0]

#                 if confi > 0.50:
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f"{class_name}: {confi:.2f}"
#                     cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # Crop the detected area from the frame
#                     cropped_frame = frame[y1:y2, x1:x2]

#                     # Use EasyOCR to extract text from the cropped area
#                     result = reader.readtext(cropped_frame)

#                     # Draw extracted text on the frame
#                     for (bbox, text, prob) in result:
#                         if prob > 0.5:  # Filter out weak predictions
#                             top_left = tuple(map(int, bbox[0]))
#                             bottom_right = tuple(map(int, bbox[2]))
#                             cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
#                             cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                             # frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#                             # cv2.imwrite(frame_filename, frame)
#                             print("text...........",text)
#         re = cv2.resize(frame, (800, 800))
#         cv2.imshow("frames", re)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#             cv2.imwrite(frame_filename, frame)

#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage

# video_path = r"D:\vChanel\spinning mill\gate entrance 28.08.24.mp4"
# yolo_model_path = r"D:\vChanel\spinning mill\t_shirt.pt"
# output_video_path = r'tshirt'

# detect_and_display_elephants(video_path, yolo_model_path, output_video_path)
# --------------------------------------------------------------

# import cv2
# from ultralytics import YOLO
# import os
# import easyocr

# def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
#     model = YOLO(yolo_model_path)
#     reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader with English language support

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = r'D:\vChanel\spinning mill\gate_entrence1'
#     os.makedirs(output_folder, exist_ok=True)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     frame_index = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_index += 1

#         yolomodel = model(frame)

#         for output in yolomodel:
#             for detection in output.boxes:
#                 confi = detection.conf[0]
#                 class_name = model.names[0]

#                 if confi > 0.50:
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f"{class_name}: {confi:.2f}"
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # Crop the detected area from the frame
#                     cropped_frame = frame[y1:y2, x1:x2]

#                     # Use EasyOCR to extract text from the cropped area
#                     result = reader.readtext(cropped_frame)


#                     # Draw extracted text on the frame
#                     for (bbox, text, prob) in result:
#                         if prob > 0.5:  # Filter out weak predictions
#                             top_left = tuple(map(int, bbox[0]))
#                             bottom_right = tuple(map(int, bbox[2]))

#                             # Adjust the text position to be above the bounding box
#                             text_position = (x1, y1 - 20)

#                             cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
#                             cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
#                             print("Text:", text)

#         re = cv2.resize(frame, (800, 800))
#         cv2.imshow("frames", re)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#             cv2.imwrite(frame_filename, frame)

#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_path = r"D:\vChanel\spinning mill\gate entrance 28.08.24.mp4"
# yolo_model_path = r"D:\vChanel\spinning mill\t_shirt.pt"
# output_video_path = r'tshirt'

# detect_and_display_elephants(video_path, yolo_model_path, output_video_path)

# import cv2
# from ultralytics import YOLO
# import os
# import easyocr
# import re  # Import regular expressions module

# def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
#     model = YOLO(yolo_model_path)
#     reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader with English language support

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = r'D:\vChanel\spinning mill\gate_entrence1'
#     os.makedirs(output_folder, exist_ok=True)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     frame_index = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_index += 1

#         yolomodel = model(frame)

#         for output in yolomodel:
#             for detection in output.boxes:
#                 confi = detection.conf[0]
#                 class_name = model.names[0]

#                 if confi > 0.50:
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f"{class_name}: {confi:.2f}"
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # Crop the detected area from the frame
#                     cropped_frame = frame[y1:y2, x1:x2]

#                     # Use EasyOCR to extract text from the cropped area
#                     result = reader.readtext(cropped_frame)

#                     # Draw extracted text on the frame
#                     for (bbox, text, prob) in result:
#                         if prob > 0.5:  # Filter out weak predictions
#                             # Check if the extracted text is an integer
#                             if re.match(r'^\d+$', text):
#                                 top_left = tuple(map(int, bbox[0]))
#                                 bottom_right = tuple(map(int, bbox[2]))

#                                 # Adjust the text position to be above the bounding box
#                                 text_position = (x1, y1 - 20)

#                                 cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
#                                 cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
#                                 print("Text:", text)

#         resized_frame = cv2.resize(frame, (800, 800))  # Renamed the variable from 're' to 'resized_frame'
#         cv2.imshow("frames", resized_frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#             cv2.imwrite(frame_filename, frame)

#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# video_path = r"D:\vChanel\spinning mill\gate entrance 28.08.24.mp4"
# yolo_model_path = r"D:\vChanel\spinning mill\t_shirt.pt"
# output_video_path = r'tshirt'

# detect_and_display_elephants(video_path, yolo_model_path, output_video_path)


import cv2
from ultralytics import YOLO
import os
import easyocr
import re  # Import regular expressions module

def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
    model = YOLO(yolo_model_path)
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader with English language support

    # Open the video
    cap = cv2.VideoCapture(video_path)
    output_folder = r'D:\vChanel\spinning mill\gate_entrence1'
    os.makedirs(output_folder, exist_ok=True)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define the detection zone (top-left and bottom-right corners)
    zone_top_left = (650, 396)  # Modify these coordinates as needed
    zone_bottom_right = (1536, 921)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Draw the detection zone on the frame
        cv2.rectangle(frame, zone_top_left, zone_bottom_right, (0, 255, 255), 2)

        yolomodel = model(frame)

        for output in yolomodel:
            for detection in output.boxes:
                confi = detection.conf[0]
                class_name = model.names[0]

                if confi > 0.50:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])

                    # Check if the detection is within the defined zone
                    if (zone_top_left[0] <= x1 <= zone_bottom_right[0] and
                        zone_top_left[1] <= y1 <= zone_bottom_right[1] and
                        zone_top_left[0] <= x2 <= zone_bottom_right[0] and
                        zone_top_left[1] <= y2 <= zone_bottom_right[1]):

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {confi:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        # Crop the detected area from the frame
                        cropped_frame = frame[y1:y2, x1:x2]

                        # Use EasyOCR to extract text from the cropped area
                        result = reader.readtext(cropped_frame)

                        # Draw extracted text on the frame
                        for (bbox, text, prob) in result:
                            if prob > 0.5:  # Filter out weak predictions
                                # Check if the extracted text is an integer
                                if re.match(r'^\d+$', text):
                                    top_left = tuple(map(int, bbox[0]))
                                    bottom_right = tuple(map(int, bbox[2]))

                                    # Adjust the text position to be above the bounding box
                                    text_position = (x1, y1 - 20)

                                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                                    print("Text:", text)

        resized_frame = cv2.resize(frame, (800, 800))  # Renamed the variable from 're' to 'resized_frame'
        cv2.imshow("frames", resized_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = r"D:\vChanel\spinning mill\gate entrance 28.08.24.mp4"
yolo_model_path = r"D:\vChanel\spinning mill\t_shirt.pt"
output_video_path = r'tshirt'

detect_and_display_elephants(video_path, yolo_model_path, output_video_path)

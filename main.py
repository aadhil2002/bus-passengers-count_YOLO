import cv2
from tracker import ObjectCounter

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(f"Mouse moved to: {point}")

# Open the video file
cap = cv2.VideoCapture('bus_video.mp4')

# Get video properties for the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
output_filename = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi format
out = cv2.VideoWriter(output_filename, fourcc, fps//2, (1020, 500))  # Note: using resized dimensions

# Define region points for counting
region_points = [(580,460), (509, 122)]

# Initialize the object counter
counter = ObjectCounter(
    region=region_points,
    model="yolo11s.pt",
    classes=[0],
    show_in=True,
    show_out=True,
    line_width=2,
)

# Create a named window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 2 != 0:  # Skip odd frames
        continue
        
    frame = cv2.resize(frame, (1020, 500))
    # Process the frame with the object counter
    frame = counter.count(frame)
    
    # Write the frame to output video
    out.write(frame)
    
    # Show the frame
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
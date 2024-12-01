import cv2
import numpy as np

def draw_grid(frame):
    height, width, _ = frame.shape
    rows, cols = 3, 3
    side_length = min(height, width) // 2

    x_start = (width - side_length) // 2
    y_start = (height - side_length) // 2

    cell_size = side_length // rows

    for i in range(rows):
        for j in range(cols):
            top_left = (x_start + j * cell_size, y_start + i * cell_size)
            bottom_right = (top_left[0] + cell_size, top_left[1] + cell_size)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 4)

def get_color(box):
    hsv_box = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv_box, axis=(0, 1))
    return avg_color

COLOR_DICT = {
    "green": 0,
    "red": 1,
    "blue": 2,
    "orange": 3,
    "white": 4,
    "yellow": 5,
    "unknown": -1,
}

COLOR_BOUNDS = {
    "white": ((0, 0, 200), (180, 50, 255)),   
    "yellow": ((20, 100, 100), (35, 255, 255)),
    "orange": ((3, 100, 100), (10, 255, 255)), 
    "red": ((170, 100, 100), (180, 255, 255)),
    "red": ((0, 100, 100), (3, 255, 255)), 
    "blue": ((100, 90, 75), (130, 255, 255)), 
    "green": ((40, 50, 50), (80, 255, 255)), 
}

NEXT_FACE = {
    0: "White on Top, Green in Front",
    1: "White on Top, Red in Front",
    2: "White on Top, Blue in Front",
    3: "White on Top, Orange in Front",
    4: "Green on Top, White in Front",
    5: "Green on Top, Yellow in Front"
}

def preprocess_region(region):
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return hsv_region

def match_color(hsv_color):
    best_match = None
    best_distance = float('inf')
    
    for color, (lower, upper) in COLOR_BOUNDS.items():
        if all(lower[i] <= hsv_color[i] <= upper[i] for i in range(3)):
            return color
        
        range_center = [(lower[i] + upper[i]) / 2 for i in range(3)]
        distance = np.linalg.norm(hsv_color - range_center)
        
        if distance < best_distance:
            best_distance = distance
            best_match = color
    
    return best_match if best_match else 'unknown'


def get_dominant_color(region):
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv_region.reshape(-1, 3)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    k = 3 
    _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    unique, counts = np.unique(labels, return_counts=True)
    dominant_center = centers[unique[np.argmax(counts)]]
    
    return dominant_center

def get_face(frame, next_face_idx):
    print("Getting face colors:")
    height, width, _ = frame.shape
    rows, cols = 3, 3
    face = np.zeros((rows, cols))
    side_length = min(height, width) // 2

    x_start = (width - side_length) // 2
    y_start = (height - side_length) // 2

    cell_size = side_length // rows
    color_array = []

    for i in range(rows):
        for j in range(cols):
            region_x_start = x_start + j * cell_size
            region_x_end = x_start + (j + 1) * cell_size
            region_y_start = y_start + i * cell_size
            region_y_end = y_start + (i + 1) * cell_size

            region = frame[region_y_start : region_y_end, region_x_start : region_x_end]
            found_color = get_dominant_color(region)            
            matched_color = match_color(found_color)
            # print(f"value: {found_color} matched to: {matched_color}")
            face[i, j] = COLOR_DICT[matched_color]
            color_array.append(matched_color)

    for i in range(3):
        print(f"------------------------")
        print(f"| {color_array[3 * i]} | {color_array[3 * i + 1]} | {color_array[3 * i + 2]} |")
    print(f"------------------------")

    if -1 in face or face[1, 1] != next_face_idx:
        print(f"Error in collecting face information!")
        return None, False
    else:
        return face, True


def get_state():
    '''
    Takes livestream of video, applies grid to it and queries to show certain faces (with specificed orientation).
    Based on the face captured, determines state of the cube (which stickers are where), and stores this to
    a 6x3x3 np array, which is returned
    '''

    cap = cv2.VideoCapture(0)
    next_face_idx = 0
    cube_state = np.zeros((6, 3, 3))

    while next_face_idx < 6:
        ret, frame = cap.read()

        if not ret:
            break

        next_required = NEXT_FACE[next_face_idx]
        h, w, _ = frame.shape
        draw_grid(frame)
        show_frame = cv2.putText(frame, f"Orient {next_required} and press C to capture state:", (h//10, w//20), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2)
        cv2.imshow("Capture Cube", show_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            arr, successful = get_face(frame, next_face_idx)
            if successful:
                cube_state[next_face_idx] = arr
                next_face_idx += 1

        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return cube_state


if __name__ == "__main__":
    cs = get_state()
    print(f"Cube State found: \n{cs}")
import cv2
import numpy as np
import os


class PolygonDefiner:
    def __init__(self):
        self.current_points_list = []
        self.frame_display = None
        self.original_frame = None
        self.window_name = ""

    def draw_polygon_callback(self, event, x, y, flags, param):
        if self.frame_display is None:
            return 

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points_list.append((x, y))
            
            cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1) 
            if len(self.current_points_list) > 1:
                cv2.line(self.frame_display, self.current_points_list[-2], self.current_points_list[-1], (255, 255, 0), 2) 
            
            if len(self.current_points_list) >= 3:
                 cv2.line(self.frame_display, self.current_points_list[-1], self.current_points_list[0], (0, 165, 255), 1) 

            cv2.imshow(self.window_name, self.frame_display)

    def define_polygon_interactive(self, approach_name, video_path):
        video_filename = os.path.basename(video_path)
        self.window_name = f"Define Polygon for {approach_name} ({video_filename})"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Polygon] Error: Could not open video file: {video_path}")
            return None

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[Polygon] Error: Could not read first frame from: {video_path}")
            cap.release()
            return None

        self.original_frame = frame.copy()
        self.frame_display = frame.copy()
        self.current_points_list = [] 

        font = cv2.FONT_HERSHEY_SIMPLEX
        instructions1 = "LClick=Add Point | RClick/u=Undo | r=Reset"
        instructions2 = "Enter/d=Save ROI (Full Frame if 0 pts) | Esc/q=Skip"
        y0, dy = 30, 30
        
        cv2.putText(self.frame_display, instructions1, (10, y0), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA) 
        cv2.putText(self.frame_display, instructions1, (10, y0), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA) 
        cv2.putText(self.frame_display, instructions2, (10, y0 + dy), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(self.frame_display, instructions2, (10, y0 + dy), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_polygon_callback) 

        print(f"\n--- Defining polygon for approach '{approach_name}' ({video_filename}) ---")
        print(f" INSTRUCTIONS: {instructions1} | {instructions2}")

        while True:
            cv2.imshow(self.window_name, self.frame_display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('d') or key == 13:
                if len(self.current_points_list) >= 3:
                    print(f" -> Polygon defined for {approach_name} with {self.current_points_list}")
                    cap.release()
                    cv2.destroyWindow(self.window_name)
                    polygon_array = np.array(self.current_points_list, dtype=np.int32) 
                    return polygon_array
                elif len(self.current_points_list) == 0:
                    h, w = frame.shape[:2]
                    default_polygon = [(0, 0), (w, 0), (w, h), (0, h)]
                    print(f" -> No custom points selected for {approach_name}. Defaulting to full frame ROI.")
                    cap.release()
                    cv2.destroyWindow(self.window_name)
                    return np.array(default_polygon, dtype=np.int32)
                else:
                    print(" -> Warning: Need at least 3 points to define a polygon (or reset with 'r' to use full frame).")

            elif key == ord('r'):
                print(" -> Resetting points.")
                self.current_points_list.clear()
                self.frame_display = self.original_frame.copy()
                cv2.putText(self.frame_display, instructions1, (10, y0), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(self.frame_display, instructions1, (10, y0), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(self.frame_display, instructions2, (10, y0 + dy), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(self.frame_display, instructions2, (10, y0 + dy), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            elif key == ord('q') or key == 27:
                print(f" -> Skipped defining polygon for {approach_name}")
                cap.release()
                cv2.destroyWindow(self.window_name)
                return None

            elif key == ord('u'):
                if self.current_points_list:
                     print(" -> Undoing last point.")
                     self.current_points_list.pop()
                     
                     self.frame_display = self.original_frame.copy()
                     cv2.putText(self.frame_display, instructions1, (10, y0), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                     cv2.putText(self.frame_display, instructions1, (10, y0), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                     cv2.putText(self.frame_display, instructions2, (10, y0 + dy), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                     cv2.putText(self.frame_display, instructions2, (10, y0 + dy), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                     
                     for i, point in enumerate(self.current_points_list):
                          cv2.circle(self.frame_display, point, 5, (0, 255, 0), -1)
                          if i > 0:
                              cv2.line(self.frame_display, self.current_points_list[i-1], point, (255, 255, 0), 2)
                     if len(self.current_points_list) >= 3:
                          cv2.line(self.frame_display, self.current_points_list[-1], self.current_points_list[0], (0, 165, 255), 1)
                else:
                     print(" -> No points to undo.")

        cap.release()
        cv2.destroyWindow(self.window_name)
        return None

def define_polygon_interactive(approach_name, video_path):
    definer = PolygonDefiner()
    return definer.define_polygon_interactive(approach_name, video_path)